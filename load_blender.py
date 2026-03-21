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


def load_blender_data(basedir, half_res=False, testskip=1, sr_basedir=None):
    """
    加载 Blender 合成数据集。
    sr_basedir: 若不为 None，就从 {sr_basedir}/{split}_sr/ 读取超分辨率图像；
                SR 图像是 RGB（无 alpha），读取后自动补全 1 的 alpha 通道保持返回格式一致。
    """
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        # 确定该 split 对应的 SR 目录（若有）
        sr_split_dir = None
        if sr_basedir is not None:
            candidate = os.path.join(sr_basedir, s + '_sr')
            if os.path.isdir(candidate) and len(os.listdir(candidate)) > 0:
                sr_split_dir = candidate
                print(f'[Blender] {s}: 使用 SR 图像目录 {sr_split_dir}')

        for frame in meta['frames'][::skip]:
            # 尝试读取 SR 图像
            loaded = False
            if sr_split_dir is not None:
                rel = frame['file_path'].replace('\\', '/').split('/')
                rel = [p for p in rel if p and p != '.']
                img_name = rel[-1] + '.png'  # e.g. 'r_0.png'
                sr_fname = os.path.join(sr_split_dir, img_name)
                if os.path.exists(sr_fname):
                    raw = imageio.imread(sr_fname)  # RGB, uint8
                    # 补全 alpha 通道（=255）保持 RGBA 格式一致
                    if raw.ndim == 3 and raw.shape[2] == 3:
                        alpha = np.full((*raw.shape[:2], 1), 255, dtype=raw.dtype)
                        raw = np.concatenate([raw, alpha], axis=-1)
                    imgs.append(raw)
                    loaded = True
            if not loaded:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res and sr_basedir is None:
        # SR 模式下不降采样，直接保留 SR 高分辨率
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        
    return imgs, poses, render_poses, [H, W, focal], i_split


