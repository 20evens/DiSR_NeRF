"""
utils.py - 工具函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from PIL import Image


def save_checkpoint(model, optimizer, epoch, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': model.config.__dict__ if hasattr(model, 'config') else {}
    }
    torch.save(checkpoint, save_path)
    print(f"检查点保存到: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    return model, optimizer, epoch


def calculate_psnr(img1, img2):
    """计算PSNR"""
    # 确保图像在0-1范围内
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    # 计算PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_ssim(img1, img2):
    """计算SSIM"""
    # 确保图像在0-1范围内
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    # 对于彩色图像，计算每个通道的平均SSIM
    if img1.shape[-1] == 3:
        ssim_values = []
        for i in range(3):
            ssim_val = structural_similarity(
                img1[..., i], img2[..., i],
                data_range=1.0, win_size=11, gaussian_weights=True
            )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return structural_similarity(
            img1, img2, data_range=1.0, win_size=11, gaussian_weights=True
        )


def prepare_data_directory(data_root, download_sample=False):
    """准备数据目录结构"""
    import shutil

    # 创建目录
    directories = ['train', 'val', 'test']
    for dir_name in directories:
        os.makedirs(os.path.join(data_root, dir_name), exist_ok=True)

    # 如果下载样本数据
    if download_sample:
        print("下载样本数据...")
        # 这里可以添加下载代码，例如从URL下载样本数据集
        # 由于需要网络连接，这里只创建占位符
        create_sample_images(data_root)


def create_sample_images(data_root):
    """创建样本图像（用于测试）"""
    from PIL import ImageDraw

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)

        for i in range(10 if split == 'train' else 5):
            # 创建简单图像
            img = Image.new('RGB', (256, 256), color='white')
            draw = ImageDraw.Draw(img)

            # 绘制一些形状
            draw.rectangle([50, 50, 150, 150], fill='red')
            draw.ellipse([150, 150, 250, 250], fill='blue')
            draw.text((100, 100), f"Sample {i + 1}", fill='black')

            # 保存图像
            img.save(os.path.join(split_dir, f"sample_{i + 1}.png"))

    print(f"样本图像已创建在 {data_root}")


def tensor_to_image(tensor):
    """将张量转换为PIL图像"""
    # 假设tensor是CHW格式，值在[-1, 1]范围内
    tensor = tensor.cpu().detach()

    # 反归一化
    tensor = tensor * 0.5 + 0.5

    # 转换为numpy并调整维度
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    numpy_image = tensor.numpy().transpose(1, 2, 0)
    numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(numpy_image)


def save_image_grid(images, filename, nrow=4):
    """保存图像网格"""
    from torchvision.utils import make_grid

    # 创建网格
    grid = make_grid(images, nrow=nrow, normalize=True, scale_each=True)

    # 转换为PIL图像
    grid_image = tensor_to_image(grid)

    # 保存
    grid_image.save(filename)
    print(f"图像网格保存到: {filename}")