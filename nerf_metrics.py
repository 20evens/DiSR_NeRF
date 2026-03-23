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

import warnings
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

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
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

def _get_niqe_fn():
    """
    获取可用的无参考图像质量计算函数，按优先级尝试多种实现：
      1. pyiqa NIQE
      2. piq.niqe（函数式）/ piq.NIQE（类式）
      3. piq.brisque 作为替代无参考指标
    返回 (fn, metric_name) 或 (None, None)
    """
    # 方案1: pyiqa NIQE（最可靠）
    try:
        import pyiqa
        niqe_model = pyiqa.create_metric('niqe', device='cpu')
        print(f'[Metrics] 使用 pyiqa 计算 NIQE')
        return lambda t: niqe_model(t).item(), 'NIQE'
    except ImportError:
        pass
    except Exception as e:
        print(f'[Metrics] pyiqa NIQE 初始化失败: {e}')

    # 方案2: piq NIQE
    try:
        import piq
        if hasattr(piq, 'niqe'):
            print(f'[Metrics] 使用 piq {piq.__version__} 函数式 API 计算 NIQE')
            return lambda t: piq.niqe(t, data_range=1.0).item(), 'NIQE'
        if hasattr(piq, 'NIQE'):
            niqe_metric = piq.NIQE(data_range=1.0)
            print(f'[Metrics] 使用 piq {piq.__version__} 类式 API 计算 NIQE')
            return lambda t: niqe_metric(t).item(), 'NIQE'
    except ImportError:
        pass

    # 方案3: piq BRISQUE 作为替代无参考指标
    try:
        import piq
        if hasattr(piq, 'brisque'):
            print(f'[Metrics] piq {piq.__version__} 不含 NIQE，使用 BRISQUE 作为替代无参考指标')
            return lambda t: piq.brisque(t, data_range=1.0).item(), 'BRISQUE'
        if hasattr(piq, 'BRISQUELoss'):
            brisque_metric = piq.BRISQUELoss(data_range=1.0)
            print(f'[Metrics] piq {piq.__version__} 不含 NIQE，使用 BRISQUELoss 作为替代无参考指标')
            return lambda t: brisque_metric(t).item(), 'BRISQUE'
    except ImportError:
        pass

    print('[Metrics] 无可用的无参考图像质量指标（NIQE/BRISQUE），跳过计算。')
    print('[Metrics] 建议运行: pip install pyiqa')
    return None, None


def compute_niqe_batch(imgs):
    """
    计算批量图像的平均无参考图像质量分数（NIQE 或 BRISQUE）。

    Args:
        imgs: list of np.ndarray [H, W, 3], float32, [0, 1]

    Returns:
        (float, str): (平均分数, 指标名称)，越低越好；若无可用实现返回 (None, None)
    """
    niqe_fn, metric_name = _get_niqe_fn()
    if niqe_fn is None:
        return None, None

    scores = []
    with torch.no_grad():
        for img in imgs:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0)
            try:
                score = niqe_fn(t)
                scores.append(score)
            except Exception as e:
                print(f'[Metrics] {metric_name} 单张计算异常（已跳过）: {e}')

    mean_score = float(np.mean(scores)) if scores else None
    return mean_score, metric_name


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

    # ── 无参考图像质量（NIQE 或 BRISQUE） ────────────────────
    print('[Metrics] 计算无参考图像质量指标...')
    niqe_score, nr_metric_name = compute_niqe_batch(imgs_pred)
    if nr_metric_name is None:
        nr_metric_name = 'NIQE'

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
        print(f'  LPIPS    (感知一致性，↓ 越低越好): {lpips_score:.4f}')
    else:
        print('  LPIPS:   未计算（无参考图像）')
    if niqe_score is not None:
        print(f'  {nr_metric_name:<8s} (无参考图像质量，↓ 越低越好): {niqe_score:.4f}')
    else:
        print(f'  {nr_metric_name}:  未计算')
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

            f.write(f'[{nr_metric_name}] 无参考图像质量（越低越好）\n')
            if niqe_score is not None:
                f.write(f'  值: {niqe_score:.6f}\n')
                f.write('  说明: 反映生成图像细节对人类视觉感知的可接受程度，无需参考图像。\n')
            else:
                f.write('  未计算\n')

        print(f'[Metrics] 指标已保存至: {save_path}')

    return {'lpips': lpips_score, nr_metric_name.lower(): niqe_score}
