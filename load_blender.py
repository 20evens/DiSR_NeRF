import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, sr_basedir=None, white_bkgd=True):
    """
    加载 Blender 合成数据集。
    内存优化：预分配 (N, H, W, 3) 输出数组，逐帧加载并立即进行 alpha 合成，
    避免创建完整 RGBA 大数组（峰值内存从 ~9 GiB 降至 ~4 GiB）。

    sr_basedir:  若不为 None，从 {sr_basedir}/{split}_sr/ 读取超分辨率图像。
    white_bkgd:  True 时对 RGBA 图像应用白色背景合成，False 时直接丢弃 alpha 通道。
    """
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # ── 第一步：统计帧数并确定 SR 目录 ────────────────────────────────
    counts = [0]
    sr_dirs = {}
    skips  = {}
    for s in splits:
        sk = 1 if (s == 'train' or testskip == 0) else testskip
        skips[s] = sk
        counts.append(counts[-1] + len(metas[s]['frames'][::sk]))
        sr_dirs[s] = None
        if sr_basedir is not None:
            candidate = os.path.join(sr_basedir, s + '_sr')
            if os.path.isdir(candidate) and len(os.listdir(candidate)) > 0:
                sr_dirs[s] = candidate
                print(f'[Blender] {s}: 使用 SR 图像目录 {candidate}')

    total_n = counts[-1]

    # ── 第二步：探测第一张图像确定 H, W ──────────────────────────────
    first_s    = 'train'
    first_frame = metas[first_s]['frames'][0]
    if sr_dirs[first_s] is not None:
        rel      = [p for p in first_frame['file_path'].replace('\\', '/').split('/') if p and p != '.']
        sr_probe = os.path.join(sr_dirs[first_s], rel[-1] + '.png')
        probe    = imageio.imread(sr_probe) if os.path.exists(sr_probe) else \
                   imageio.imread(os.path.join(basedir, first_frame['file_path'] + '.png'))
    else:
        probe = imageio.imread(os.path.join(basedir, first_frame['file_path'] + '.png'))

    H_full, W_full = probe.shape[:2]
    if half_res and sr_basedir is None:
        H_out, W_out = H_full // 2, W_full // 2
    else:
        H_out, W_out = H_full, W_full

    # ── 第三步：预分配 (N, H, W, 3) 输出数组 ─────────────────────────
    # 峰值内存仅为该数组大小 + 单帧临时数组，彻底避免存储完整 RGBA 数组
    imgs_all  = np.empty((total_n, H_out, W_out, 3), dtype=np.float32)
    poses_all = np.empty((total_n, 4, 4),             dtype=np.float32)

    idx = 0
    for s in splits:
        sr_split_dir = sr_dirs[s]
        sk           = skips[s]

        for frame in metas[s]['frames'][::sk]:
            # 读取原始图像（uint8）
            raw    = None
            loaded = False
            if sr_split_dir is not None:
                rel    = [p for p in frame['file_path'].replace('\\', '/').split('/') if p and p != '.']
                sr_fname = os.path.join(sr_split_dir, rel[-1] + '.png')
                if os.path.exists(sr_fname):
                    raw    = imageio.imread(sr_fname)
                    loaded = True
            if not loaded:
                raw = imageio.imread(os.path.join(basedir, frame['file_path'] + '.png'))

            # 转换为 float32 [0, 1]
            img_f = raw.astype(np.float32) / 255.0

            # Alpha 合成（逐帧处理，不创建整体临时数组）
            if img_f.ndim == 3 and img_f.shape[2] == 4:
                if white_bkgd:
                    a       = img_f[..., 3:4]
                    img_rgb = img_f[..., :3] * a + (1.0 - a)
                else:
                    img_rgb = img_f[..., :3]
            else:
                img_rgb = img_f[..., :3]

            # half_res 缩放（仅非 SR 模式）
            if half_res and sr_basedir is None:
                img_rgb = cv2.resize(img_rgb, (W_out, H_out), interpolation=cv2.INTER_AREA)

            imgs_all[idx]  = img_rgb
            poses_all[idx] = np.array(frame['transform_matrix'])
            idx += 1

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = 0.5 * W_full / np.tan(0.5 * camera_angle_x)
    if half_res and sr_basedir is None:
        focal /= 2.0

    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0
    )

    return imgs_all, poses_all, render_poses, [H_out, W_out, focal], i_split


