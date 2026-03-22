"""
test_full.py - 对完整测试图像进行超分辨率重建
将 800×800 低分辨率图像恢复为 1600×1600 高分辨率图像（2x SR）

处理流程：
  800×800 LR
    → 切成 128×128 小块（50% 重叠，stride=64）
    → 扩散模型每块 128×128 → 256×256（2x 学习超分）
    → Gaussian 加权混合拼接为 1600×1600（800×2）
    → 可选双三次插值 2x → 3200×3200（最终4x目标）

用法：
  python test_full.py                          # 使用 best_model.pth（默认），2x输出
  python test_full.py --checkpoint ./checkpoints/latest.pth
  python test_full.py --bicubic                 # 额外2x双三次插值得到4x输出
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import math
import glob
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

from config import Config
from model import DiffusionModel
from utils import load_checkpoint, calculate_psnr, calculate_ssim

# GPU 极限优化配置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')


def color_match(sr_np, lr_np):
    """
    频域颜色校正：从SR提取高频纹理细节，从双三次LR锚定低频颜色。
    消除扩散模型引入的颜色偏差，同时保留SR的锐利细节。
    sr_np: [H_sr, W_sr, 3] float32 [0,1]
    lr_np: [H_lr, W_lr, 3] float32 [0,1]
    """
    h, w = sr_np.shape[:2]
    lr_pil = Image.fromarray((np.clip(lr_np, 0, 1) * 255).astype(np.uint8))
    lr_up = np.array(lr_pil.resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0

    try:
        from scipy.ndimage import gaussian_filter
        # 高斯滤波提取SR的低频分量（携带颜色）
        sigma = 3.0
        sr_low = gaussian_filter(sr_np, sigma=[sigma, sigma, 0])
        # SR高频细节（纹理、锐度）
        detail = sr_np - sr_low
        # 最终 = 双三次LR（颜色正确）+ SR高频细节（锐利纹理）
        result = lr_up + detail
    except ImportError:
        # 回退：均值方差匹配
        result = np.zeros_like(sr_np)
        for c in range(3):
            sr_ch = sr_np[:, :, c]
            lr_ch = lr_up[:, :, c]
            sr_mean, sr_std = sr_ch.mean(), sr_ch.std() + 1e-6
            lr_mean, lr_std = lr_ch.mean(), lr_ch.std() + 1e-6
            result[:, :, c] = (sr_ch - sr_mean) * (lr_std / sr_std) + lr_mean

    return np.clip(result, 0, 1)


def denoise_sr(sr_np):
    """
    后处理去噪：去除扩散模型引入的高频噪点，保留边缘细节。
    优先使用OpenCV双边滤波，其次scipy高斯滤波。
    sr_np: [H, W, 3] float32 [0,1]
    """
    img_u8 = (np.clip(sr_np, 0, 1) * 255).astype(np.uint8)
    try:
        import cv2
        # 双边滤波：保边去噪，d=7邻域，sigmaColor/Space=50
        denoised = cv2.bilateralFilter(img_u8, d=7, sigmaColor=50, sigmaSpace=50)
        return denoised.astype(np.float32) / 255.0
    except ImportError:
        pass
    try:
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(sr_np, sigma=[0.6, 0.6, 0])
        return np.clip(result, 0, 1)
    except ImportError:
        return sr_np


def _gaussian_window_2d(size, sigma=None):
    """生成2D高斯权重窗口，用于重叠patch边界平滑混合。"""
    if sigma is None:
        sigma = size / 4.0
    coords = torch.arange(size, dtype=torch.float32) - size / 2.0
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g2d = torch.outer(g1d, g1d)
    return (g2d / g2d.max()).float()


def extract_patches_for_image(lr_tensor, config):
    """
    从LR图像提取所有重叠patch，返回(patches_cpu, meta)。
    patches_cpu: [N, C, patch_lr, patch_lr] CPU tensor
    meta: 结果组装所需的尺寸和位置信息
    """
    patch_lr  = config.lr_size
    patch_hr  = config.hr_size
    scale     = config.scale_factor
    stride_lr = patch_lr * 3 // 4  # 25% 重叠（vs 原50%）：patch总数减少～2x，持续圆滑混合补偿边界

    _, C, H_orig, W_orig = lr_tensor.shape
    half   = patch_lr // 2
    padded = F.pad(lr_tensor, (half, half, half, half), mode='reflect')
    _, C, H_pad, W_pad = padded.shape

    n_h = max(1, math.ceil((H_pad - patch_lr) / stride_lr) + 1)
    n_w = max(1, math.ceil((W_pad - patch_lr) / stride_lr) + 1)

    patches_list = []
    positions_hr = []
    for i in range(n_h):
        for j in range(n_w):
            y0 = min(i * stride_lr, H_pad - patch_lr)
            x0 = min(j * stride_lr, W_pad - patch_lr)
            patch = padded[:, :, y0:y0+patch_lr, x0:x0+patch_lr].squeeze(0)
            patches_list.append(patch)
            positions_hr.append((y0 * scale, x0 * scale))

    patches = torch.stack(patches_list, dim=0)  # [N, C, lr_size, lr_size] CPU tensor
    meta = {
        'H_orig': H_orig, 'W_orig': W_orig,
        'H_pad':  H_pad,  'W_pad':  W_pad,
        'C': C, 'scale': scale, 'patch_hr': patch_hr,
        'half': half, 'positions_hr': positions_hr,
    }
    return patches, meta


def assemble_sr_image(sr_patches_t, meta, device):
    """
    将SR patches用Gaussian加权混合组装回完整SR图像。
    sr_patches_t: [N, C, patch_hr, patch_hr] GPU tensor
    返回: [1, C, H_orig*scale, W_orig*scale] CPU tensor
    """
    C, scale = meta['C'], meta['scale']
    H_orig, W_orig = meta['H_orig'], meta['W_orig']
    H_pad, W_pad   = meta['H_pad'],  meta['W_pad']
    patch_hr = meta['patch_hr']
    half     = meta['half']
    positions_hr = meta['positions_hr']

    H_sr, W_sr = H_pad * scale, W_pad * scale
    gauss   = _gaussian_window_2d(patch_hr).to(device)
    gauss_c = gauss.unsqueeze(0).expand(C, -1, -1).contiguous()

    sr_accum = torch.zeros(1, C, H_sr, W_sr, device=device)
    w_accum  = torch.zeros(1, 1, H_sr, W_sr, device=device)

    for idx, (py, px) in enumerate(positions_hr):
        sr_accum[0, :, py:py+patch_hr, px:px+patch_hr].add_(sr_patches_t[idx] * gauss_c)
        w_accum[0, 0, py:py+patch_hr, px:px+patch_hr].add_(gauss)

    w_accum.clamp_(min=1e-6)
    sr_canvas = sr_accum / w_accum
    pad_sr = half * scale
    result = sr_canvas[:, :, pad_sr:pad_sr + H_orig*scale, pad_sr:pad_sr + W_orig*scale]
    return result.cpu()


def pad_to_multiple(img_tensor, multiple):
    """将图像 [1,C,H,W] 填充到 multiple 的整数倍，返回 (padded, orig_h, orig_w)"""
    _, c, h, w = img_tensor.shape
    pad_h = math.ceil(h / multiple) * multiple - h
    pad_w = math.ceil(w / multiple) * multiple - w
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return img_tensor, h, w




def sr_full_image(model, lr_tensor, config):
    """
    重叠patch超分辨率 + Gaussian加权混合，消除方块边界。
    stride = patch_lr//2 (50%重叠)，Gaussian窗口在重叠区域平滑过渡。
    lr_tensor: [1, C, H, W]，值在 [-1, 1]
    返回: SR图像 tensor [1, C, H*scale, W*scale]，值在 [-1, 1]
    """
    device = next(model.parameters()).device
    lr_tensor = lr_tensor.to(device)

    patch_lr  = config.lr_size       # 128
    patch_hr  = config.hr_size       # 256
    scale     = config.scale_factor  # 2
    stride_lr = patch_lr * 3 // 4   # 96 — 25% 重叠（毕50%少～2x patch）
    stride_hr = stride_lr * scale    # 192

    _, C, H_orig, W_orig = lr_tensor.shape

    # reflect padding：确保边界 patch 有足够上下文
    half   = patch_lr // 2   # 64（padding 与 stride 无关，保持不变）
    padded = F.pad(lr_tensor, (half, half, half, half), mode='reflect')
    _, C, H_pad, W_pad = padded.shape

    # 覆盖整个 padded 图像需要的 patch 数量
    n_h = max(1, math.ceil((H_pad - patch_lr) / stride_lr) + 1)
    n_w = max(1, math.ceil((W_pad - patch_lr) / stride_lr) + 1)
    total = n_h * n_w

    # SR 输出画布（对应 padded 图像尺寸）
    H_sr = H_pad * scale
    W_sr = W_pad * scale
    sr_accum = torch.zeros(1, C, H_sr, W_sr, device=device)
    w_accum  = torch.zeros(1, 1, H_sr, W_sr, device=device)

    # 高斯权重窗口：重叠区域平滑混合
    gauss   = _gaussian_window_2d(patch_hr).to(device)           # [hr_size, hr_size]
    gauss_c = gauss.unsqueeze(0).expand(C, -1, -1).contiguous()  # [C, hr_size, hr_size]

    # 提取所有重叠 patch，记录对应 SR 画布位置
    patches_list = []
    positions_hr = []
    for i in range(n_h):
        for j in range(n_w):
            y0 = min(i * stride_lr, H_pad - patch_lr)
            x0 = min(j * stride_lr, W_pad - patch_lr)
            patch = padded[:, :, y0:y0+patch_lr, x0:x0+patch_lr].squeeze(0)  # [C,lr,lr]
            patches_list.append(patch)
            positions_hr.append((y0 * scale, x0 * scale))

    all_patches = torch.stack(patches_list, dim=0).half()  # [total, C, lr_size, lr_size] FP16
    print(f"  重叠patch数: {total}（stride={stride_lr}，25%重叠），DDIM步数=10")

    # 批量 SR 推理（显存不足自动降批）
    BATCH = min(total, 96)   # hr=256 + 较少patch，可提大batch
    with torch.inference_mode():
        sr_parts = []
        for start in range(0, total, BATCH):
            end = min(start + BATCH, total)
            try:
                sr_parts.append(model.sample_ddim(all_patches[start:end], ddim_steps=10))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                BATCH = max(BATCH // 2, 1)
                print(f"\n  [OOM] batch_size 降至 {BATCH}")
                for s in range(start, end, BATCH):
                    e = min(s + BATCH, end)
                    sr_parts.append(model.sample_ddim(all_patches[s:e], ddim_steps=10))
            print(f"\r  patch {min(end, total):3d}/{total}", end="", flush=True)

        sr_all = torch.cat(sr_parts, dim=0)  # [total, C, hr_size, hr_size]

        # ★ Gaussian 加权混合回画布（消除方块边界）
        for idx, (py, px) in enumerate(positions_hr):
            sr_accum[0, :, py:py+patch_hr, px:px+patch_hr].add_(sr_all[idx] * gauss_c)
            w_accum[0, 0, py:py+patch_hr, px:px+patch_hr].add_(gauss)

        w_accum.clamp_(min=1e-6)
        sr_canvas = sr_accum / w_accum

    print()
    # 裁剪：去掉 padding 引入的边缘（每侧 = half * scale 像素）
    pad_sr = half * scale
    return sr_canvas[:, :, pad_sr:pad_sr + H_orig*scale, pad_sr:pad_sr + W_orig*scale]


def tensor_to_pil(tensor):
    """[1,C,H,W] [-1,1] → PIL RGB Image"""
    arr = np.clip((tensor[0].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def run_sr(input_dir, output_dir, checkpoint=None, no_bicubic=True, max_images=None):
    """
    对指定目录中的图像进行超分辨率重建。
    可以被 run_nerf.py 直接调用。
    """
    config = Config()

    # 加载模型
    model = DiffusionModel(config).to(config.device)
    ckpt = checkpoint or os.path.join(config.checkpoint_dir, "best_model.pth")
    if os.path.exists(ckpt):
        model, _, _ = load_checkpoint(model, None, ckpt, config.device)
        print(f"[SR] 加载检查点: {ckpt}")
    else:
        print(f"[SR] 错误: 检查点不存在: {ckpt}")
        return
    model.eval()

    # channels_last 内存格式：加速卷积运算 10-30%
    model = model.to(memory_format=torch.channels_last)
    # FP16 半精度：减少显存占用 + 加速推理
    model = model.half()
    print("[SR] channels_last + FP16 已启用")

    # torch.compile 加速（使用 default 模式，Windows 兼容性更好）
    if hasattr(torch, 'compile'):
        try:
            model.model = torch.compile(model.model, mode='default')
            print("[SR] torch.compile 已启用（default 模式）")
        except Exception as e:
            print(f"[SR] torch.compile 不可用: {e}")

    # 预热 GPU：第一次推理有编译开销，用小 batch 触发编译
    print("[SR] 预热 GPU...")
    with torch.inference_mode():
        _dummy = torch.randn(1, config.channels, config.lr_size, config.lr_size,
                             device=config.device, dtype=torch.half
                             ).to(memory_format=torch.channels_last)
        _ = model.sample_ddim(_dummy, ddim_steps=2)
        del _dummy, _
        torch.cuda.empty_cache()
    print("[SR] 预热完成")

    # 图像预处理
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找测试图像
    exts = ["*.png", "*.jpg", "*.JPG", "*.jpeg"]
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files = sorted(set(image_files))

    if max_images:
        image_files = image_files[:max_images]

    print(f"[SR] 共找到 {len(image_files)} 张图像，输入: {input_dir}")
    final_scale = config.scale_factor if no_bicubic else config.scale_factor * 2
    print(f"[SR] 模型倍率: {config.scale_factor}x  →  最终倍率: "
          f"{final_scale}x（{'仅SR模型' if no_bicubic else f'{config.scale_factor}x SR + 2x双三次'}）")
    print(f"[SR] 结果保存至: {output_dir}\n")

    # ═══════════════════════════════════════════════════
    # 阶段1/3：并行读取图像 + 提取所有重叠patch（CPU）
    # 使用 ThreadPoolExecutor 并发执行 I/O + patch 提取
    # ═══════════════════════════════════════════════════
    print(f"[SR] 阶段 1/3：并行读取图像 + 提取patch（{len(image_files)} 张）...")

    def _load_and_extract(img_path):
        raw_pil = Image.open(img_path)
        if raw_pil.mode == 'RGBA':
            # blender/nerf_synthetic: 透明背景合成到白色（与 white_bkgd=True 一致）
            bg = Image.new('RGB', raw_pil.size, (255, 255, 255))
            bg.paste(raw_pil, mask=raw_pil.split()[3])
            lr_pil = bg
        else:
            lr_pil = raw_pil.convert("RGB")
        lr_np     = np.array(lr_pil).astype(np.float32) / 255.0
        lr_tensor = to_tensor(lr_pil).unsqueeze(0)          # [1,3,H,W] CPU
        patches, meta = extract_patches_for_image(lr_tensor, config)
        w, h = lr_pil.size
        return img_path, lr_np, patches, meta, (w, h)

    with ThreadPoolExecutor(max_workers=min(8, len(image_files))) as pool:
        extracted = list(pool.map(_load_and_extract, image_files))

    img_infos    = []   # (img_path, lr_np, meta)
    patches_list = []   # 每张图像的 [N_i, C, lr_size, lr_size] CPU tensor
    patch_counts = []   # 每张图像的 patch 数

    for img_path, lr_np, patches, meta, (w, h) in extracted:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {name}: {w}×{h}  →  {patches.shape[0]} patches")
        img_infos.append((img_path, lr_np, meta))
        patches_list.append(patches)
        patch_counts.append(patches.shape[0])

    all_patches_cpu = torch.cat(patches_list, dim=0)   # [total, C, lr_size, lr_size]
    total_patches   = all_patches_cpu.shape[0]
    avg_per_img     = total_patches // len(image_files)
    print(f"  → 合计 {total_patches} 个patch（均 {avg_per_img}/张）\n")
    del patches_list

    # ═══════════════════════════════════════════════════
    # 阶段 2/3 + 3/3：分组流水线（每 SAVE_EVERY 张存储一次）
    # 每组：DDIM 推理 → GPU Gaussian 组装 → CPU 并行保存 → 释放显存
    # 好处：峰值内存低；无需等全部图片完成即持续写入磁盘
    # ═══════════════════════════════════════════════════
    DDIM_STEPS = 10    # 20→10：推理 2x 加速
    DDIM_BATCH = 128   # hr=256 + 25% 重叠，可用较大 batch
    SAVE_EVERY = 10    # 每处理 N 张立即保存一次

    # 预计算每张图像 patch 在 all_patches_cpu 中的起止偏移
    # cum[i] = 第 i 张图像 patch 的起始索引；cum[i+1] = 结束索引
    cum = [0]
    for n in patch_counts:
        cum.append(cum[-1] + n)

    # pin_memory：锁页内存，加速 CPU → GPU 传输
    all_patches_cpu = all_patches_cpu.pin_memory()

    n_imgs  = len(img_infos)
    saved_n = 0

    # ── 后处理 + 写磁盘（并行 CPU）──────────────────────────────
    def _postprocess_and_save(args):
        g_idx, img_path, lr_np, sr_cpu = args
        if not no_bicubic:
            _, _, h4, w4 = sr_cpu.shape
            sr_cpu = F.interpolate(
                sr_cpu, size=(h4 * 2, w4 * 2),
                mode='bicubic', align_corners=False)
        sr_np = np.clip(
            (sr_cpu[0].numpy() * 0.5 + 0.5).transpose(1, 2, 0), 0, 1)
        sr_np = color_match(sr_np, lr_np)
        sr_np = denoise_sr(sr_np)
        sr_path = os.path.join(output_dir, os.path.basename(img_path))
        Image.fromarray((sr_np * 255).astype(np.uint8)).save(sr_path)
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  [{g_idx + 1}/{n_imgs}] {name} → 已保存")

    # ── 分组主循环 ───────────────────────────────────────────────
    for g_start in range(0, n_imgs, SAVE_EVERY):
        g_end   = min(g_start + SAVE_EVERY, n_imgs)
        g_size  = g_end - g_start
        p_start = cum[g_start]
        p_end   = cum[g_end]
        g_total = p_end - p_start

        print(f"\n[SR] 阶段 2/3：图像 {g_start + 1}–{g_end}/{n_imgs}"
              f"（{g_total} patches，DDIM 步数={DDIM_STEPS}）...")

        # ── DDIM 推理（仅当前组的 patches）─────────────────────
        group_patches = all_patches_cpu[p_start:p_end]
        sr_parts      = []
        batch         = DDIM_BATCH
        s             = 0
        with torch.inference_mode():
            while s < g_total:
                e    = min(s + batch, g_total)
                bgpu = group_patches[s:e].to(
                    config.device, dtype=torch.half, non_blocking=True)
                try:
                    sr_b = model.sample_ddim(bgpu, ddim_steps=DDIM_STEPS)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch = max(batch // 2, 1)
                    print(f"\n  [OOM] batch 降至 {batch}，重试...")
                    continue    # 重试相同的 s
                sr_parts.append(sr_b.cpu())
                print(f"\r  patch {min(e, g_total):4d}/{g_total}  batch={e - s}",
                      end="", flush=True)
                s = e

        print()
        group_sr_cpu = torch.cat(sr_parts, dim=0)   # [g_total, C, hr, hr] CPU
        del sr_parts
        torch.cuda.empty_cache()

        # ── GPU Gaussian 混合组装（串行避免显存竞争）─────────────
        print(f"[SR] 阶段 3/3：组装图像 {g_start + 1}–{g_end}...")
        assembled = []
        offset    = 0
        for i in range(g_size):
            n_p        = patch_counts[g_start + i]
            sr_patches = group_sr_cpu[offset:offset + n_p].to(config.device)
            offset    += n_p
            sr_cpu     = assemble_sr_image(
                sr_patches, img_infos[g_start + i][2], config.device)
            assembled.append((g_start + i,
                               img_infos[g_start + i][0],
                               img_infos[g_start + i][1],
                               sr_cpu))
            print(f"\r  组装: {i + 1}/{g_size}", end="", flush=True)

        print()
        del group_sr_cpu
        torch.cuda.empty_cache()

        # ── 并行 CPU 后处理 + 写磁盘 ────────────────────────────
        with ThreadPoolExecutor(max_workers=min(4, g_size)) as pool:
            list(pool.map(_postprocess_and_save, assembled))

        saved_n += g_size
        del assembled
        print(f"  → 已累计保存 {saved_n}/{n_imgs} 张")

    del all_patches_cpu
    print(f"\n[SR] 完成！共处理 {len(image_files)} 张图像，结果在: {output_dir}")
    return len(image_files)


def run_sr_blender_splits(basedir, checkpoint=None, no_bicubic=True,
                          splits=None):
    """
    对 blender/nerf_synthetic 数据集按 split 分别独立进行超分辨率。
    每个 split 完整走完阶段1→阶段2→阶段3后，再处理下一个 split。
    三个 split 全部完成后返回，之后由调用方继续三维重建。

    basedir  : 数据集根目录，如 ./data/nerf_synthetic/lego
    splits   : 处理顺序列表，默认 ['train', 'val', 'test']
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    completed, skipped = [], []

    for split in splits:
        split_dir = os.path.join(basedir, split)
        sr_dir    = os.path.join(basedir, split + '_sr')

        if not os.path.isdir(split_dir):
            print(f'[SR-Blender] {split}/ 目录不存在，跳过')
            continue

        if os.path.isdir(sr_dir) and len(os.listdir(sr_dir)) > 0:
            print(f'[SR-Blender] {split}_sr/ 已存在，跳过')
            skipped.append(split)
            continue

        print(f'\n[SR-Blender] {"="*40}')
        print(f'[SR-Blender]  开始处理 split: {split}')
        print(f'[SR-Blender]  输入: {split_dir}')
        print(f'[SR-Blender]  输出: {sr_dir}')
        print(f'[SR-Blender] {"="*40}')

        run_sr(input_dir=split_dir, output_dir=sr_dir,
               checkpoint=checkpoint, no_bicubic=no_bicubic)

        completed.append(split)
        print(f'[SR-Blender] {"="*40}')
        print(f'[SR-Blender]  {split} 超分辨率完成 → {sr_dir}')
        print(f'[SR-Blender] {"="*40}\n')

    print(f'[SR-Blender] 所有 split 超分辨率完成。'
          f'已处理: {completed}  已跳过: {skipped}')
    print(f'[SR-Blender] 进入三维重建阶段...\n')


def main():
    parser = argparse.ArgumentParser(description="完整图像超分辨率重建（800→1600，2x SR）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径，默认使用 best_model.pth")
    parser.add_argument("--bicubic", action="store_true",
                        help="额外做2x双三次插值到4x结果（默认仅输出2x SR：1600×1600）")
    parser.add_argument("--max_images", type=int, default=None,
                        help="最多处理几张图像，默认处理全部")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="输入图像目录（默认使用config中的test_dir）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/sr_full_1600）")
    args = parser.parse_args()

    config = Config()
    input_dir = args.input_dir or os.path.join(config.data_root, config.test_dir)
    output_dir = args.output_dir or os.path.join(config.result_dir, "sr_full_1600")

    run_sr(input_dir, output_dir,
           checkpoint=args.checkpoint,
           no_bicubic=not args.bicubic,
           max_images=args.max_images)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
