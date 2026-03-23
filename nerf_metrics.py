"""
nerf_metrics.py - Super-NeRF 评价指标实现

参考文献: Super-NeRF: View-consistent Detail Generation for NeRF super-resolution
核心观点:
  - 由于 SR 模型生成的高分辨率细节无法保证与真实 HR 图像完全一致，
    不适合直接用 PSNR/SSIM 对比真实 HR 图像。
  - LPIPS: 感知一致性评价，反映渲染结果与参考图像在人类感知层面的相似度，越低越好。
  - NIQE:  无参考图像质量评价，反映生成图像对人类视觉感知的可接受程度，越低越好。

依赖安装（仅在首次使用时需要）:
  pip install lpips
  pip install piq
"""

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
# LPIPS
# ──────────────────────────────────────────────────────────────

def compute_lpips_batch(imgs_pred, imgs_gt, device):
    """
    计算批量图像的平均 LPIPS 感知相似度。

    Args:
        imgs_pred: list of np.ndarray [H, W, 3], float32, [0, 1]  — NeRF 渲染图像
        imgs_gt:   list of np.ndarray [H, W, 3], float32, [0, 1]  — 参考图像
        device:    torch.device 或字符串

    Returns:
        float: 平均 LPIPS（越低越好），若库未安装返回 None
    """
    try:
        import lpips as lpips_lib
    except ImportError:
        print('[Metrics] lpips 未安装，跳过 LPIPS 计算。请运行: pip install lpips')
        return None

    loss_fn = lpips_lib.LPIPS(net='alex', verbose=False).to(device)
    loss_fn.eval()

    scores = []
    with torch.no_grad():
        for pred, gt in zip(imgs_pred, imgs_gt):
            # [H, W, 3] float [0,1] → [1, 3, H, W] float [-1, 1]
            p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
            g = torch.from_numpy(gt ).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
            scores.append(loss_fn(p, g).item())

    return float(np.mean(scores)) if scores else None


# ──────────────────────────────────────────────────────────────
# NIQE
# ──────────────────────────────────────────────────────────────

def compute_niqe_batch(imgs):
    """
    计算批量图像的平均 NIQE 无参考图像质量分数。

    Args:
        imgs: list of np.ndarray [H, W, 3], float32, [0, 1]

    Returns:
        float: 平均 NIQE（越低越好），若库未安装返回 None
    """
    try:
        import piq
    except ImportError:
        print('[Metrics] piq 未安装，跳过 NIQE 计算。请运行: pip install piq')
        return None

    scores = []
    with torch.no_grad():
        for img in imgs:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0)
            try:
                score = piq.niqe(t, data_range=1.0).item()
                scores.append(score)
            except Exception as e:
                print(f'[Metrics] NIQE 单张计算异常（已跳过）: {e}')

    return float(np.mean(scores)) if scores else None


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def evaluate_nerf_metrics(rgbs, gt_imgs=None, device='cuda', save_path=None):
    """
    计算 Super-NeRF 评价指标，并可选地将结果写入 txt 文件。

    Args:
        rgbs:      np.ndarray [N, H, W, 3], float32, [0, 1]  — NeRF 渲染结果
        gt_imgs:   np.ndarray [N, H, W, 3/4] 或 None         — 参考图像（用于 LPIPS）
                   可以是原始 LR、真实 HR 或 SR 生成的图像。
                   若为 None，仅计算 NIQE（完全无参考场景）。
        device:    str 或 torch.device
        save_path: str 或 None，若指定则将指标写入该 txt 文件

    Returns:
        dict: {'lpips': float|None, 'niqe': float|None}
    """
    imgs_pred = [np.clip(rgbs[i].astype(np.float32), 0.0, 1.0) for i in range(len(rgbs))]

    # ── NIQE（无参考） ──────────────────────────────────────
    print('[Metrics] 计算 NIQE（无参考图像质量）...')
    niqe_score = compute_niqe_batch(imgs_pred)

    # ── LPIPS（需要参考图像） ───────────────────────────────
    lpips_score = None
    if gt_imgs is not None:
        print('[Metrics] 计算 LPIPS（感知一致性）...')
        gt_list = []
        for img in gt_imgs:
            img = np.clip(img.astype(np.float32), 0.0, 1.0)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]   # RGBA → RGB（Blender 数据集）
            gt_list.append(img)
        lpips_score = compute_lpips_batch(imgs_pred, gt_list, device)
    else:
        print('[Metrics] 无参考图像 → 仅使用 NIQE（符合 Super-NeRF 无真实HR评估场景）')

    # ── 打印 ───────────────────────────────────────────────
    print(f'\n{"=" * 55}')
    print(' Super-NeRF 评价指标结果')
    print(f'{"=" * 55}')
    if lpips_score is not None:
        print(f'  LPIPS  (感知一致性，↓ 越低越好): {lpips_score:.4f}')
    else:
        print('  LPIPS: 未计算（无参考图像）')
    if niqe_score is not None:
        print(f'  NIQE   (无参考图像质量，↓ 越低越好): {niqe_score:.4f}')
    else:
        print('  NIQE:  未计算')
    print(f'{"=" * 55}\n')

    # ── 保存到文件 ─────────────────────────────────────────
    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('Super-NeRF 评价指标\n')
            f.write('参考文献: Super-NeRF: View-consistent Detail Generation for NeRF super-resolution\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'渲染图像数量: {len(rgbs)}\n\n')

            f.write('[LPIPS] 感知一致性（越低越好）\n')
            if lpips_score is not None:
                f.write(f'  值: {lpips_score:.6f}\n')
                f.write('  说明: 衡量渲染结果与参考图像在人类感知层面的相似度。\n')
                f.write('        参考图像为训练所用图像（原始 LR 上采样或 SR 生成 HR）。\n\n')
            else:
                f.write('  未计算（无参考图像，符合 Super-NeRF 无真实HR评估场景）\n\n')

            f.write('[NIQE] 无参考图像质量（越低越好）\n')
            if niqe_score is not None:
                f.write(f'  值: {niqe_score:.6f}\n')
                f.write('  说明: 反映生成图像细节对人类视觉感知的可接受程度，无需参考图像。\n')
            else:
                f.write('  未计算\n')

        print(f'[Metrics] 指标已保存至: {save_path}')

    return {'lpips': lpips_score, 'niqe': niqe_score}
