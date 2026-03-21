"""
test.py - 测试训练好的模型
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 Windows 上 OpenMP 库冲突

import torch
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from config import Config
from model import DiffusionModel
from dataset import SRDataset
from utils import load_checkpoint, calculate_psnr, calculate_ssim

def test_model(config, checkpoint_path=None):
    """测试模型"""
    # 加载测试数据集
    test_dataset = SRDataset(
        os.path.join(config.data_root, config.test_dir),
        hr_size=config.hr_size,
        lr_size=config.lr_size,
        scale_factor=config.scale_factor,
        is_train=False
    )

    # 初始化模型
    model = DiffusionModel(config).to(config.device)

    # 加载检查点
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, "final_model.pth")

    if os.path.exists(checkpoint_path):
        model, _, _ = load_checkpoint(model, None, checkpoint_path, config.device)
        print(f"加载检查点: {checkpoint_path}")
    else:
        print(f"警告: 检查点不存在: {checkpoint_path}")
        return

    model.eval()

    # 测试指标
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    # 创建结果目录
    result_subdir = os.path.join(config.result_dir, "test_results")
    os.makedirs(result_subdir, exist_ok=True)

    # 测试循环
    with torch.no_grad():
        for i in range(min(20, len(test_dataset))):  # 测试前20个样本
            # 获取样本
            hr_image, lr_image = test_dataset[i]
            hr_image = hr_image.unsqueeze(0).to(config.device)
            lr_image = lr_image.unsqueeze(0).to(config.device)

            # 生成超分辨率图像
            sr_image = model.sample(lr_image)

            # 转换为numpy并clip到[0,1]范围
            hr_np = np.clip((hr_image[0].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0), 0, 1)
            lr_np = np.clip((lr_image[0].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0), 0, 1)
            sr_np = np.clip((sr_image[0].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0), 0, 1)

            # 计算指标
            psnr = calculate_psnr(hr_np, sr_np)
            ssim = calculate_ssim(hr_np, sr_np)

            total_psnr += psnr
            total_ssim += ssim
            num_samples += 1

            print(f"样本 {i + 1}: PSNR={psnr:.4f} dB, SSIM={ssim:.4f}")

            # 保存结果图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(lr_np)
            axes[0].set_title(f"低分辨率 (x{config.scale_factor})")
            axes[0].axis('off')

            axes[1].imshow(sr_np)
            axes[1].set_title(f"超分辨率 (PSNR: {psnr:.2f} dB)")
            axes[1].axis('off')

            axes[2].imshow(hr_np)
            axes[2].set_title("原始高分辨率")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(result_subdir, f"result_{i + 1}.png"), dpi=150)
            plt.close()

            # 保存单个超分辨率图像
            Image.fromarray((sr_np * 255).astype(np.uint8)).save(
                os.path.join(result_subdir, f"sr_{i + 1}.png")
            )
            # 保存对应的低分辨率和高分辨率图像
            Image.fromarray((lr_np * 255).astype(np.uint8)).save(
                os.path.join(result_subdir, f"lr_{i + 1}.png")
            )
            Image.fromarray((hr_np * 255).astype(np.uint8)).save(
                os.path.join(result_subdir, f"hr_{i + 1}.png")
            )

    # 计算平均指标
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"\n{'=' * 50}")
    print(f"测试结果 (共{num_samples}个样本):")
    print(f"平均PSNR: {avg_psnr:.4f} dB")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"{'=' * 50}")

    # 生成网格图像
    generate_sample_grid(model, test_dataset, config, result_subdir)


def generate_sample_grid(model, dataset, config, save_dir, num_samples=9):
    """生成样本网格"""
    model.eval()

    with torch.no_grad():
        # 获取样本
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        lr_images = []
        sr_images = []
        hr_images = []

        for idx in indices:
            hr, lr = dataset[idx]
            lr = lr.unsqueeze(0).to(config.device)

            # 生成超分辨率图像
            sr = model.sample(lr)

            lr_images.append(lr.cpu())
            sr_images.append(sr.cpu())
            hr_images.append(hr.unsqueeze(0))

        # 创建网格
        lr_grid = make_grid(torch.cat(lr_images), nrow=3, normalize=True, scale_each=True)
        sr_grid = make_grid(torch.cat(sr_images), nrow=3, normalize=True, scale_each=True)
        hr_grid = make_grid(torch.cat(hr_images), nrow=3, normalize=True, scale_each=True)

        # 保存网格
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        axes[0].imshow(lr_grid.permute(1, 2, 0))
        axes[0].set_title("低分辨率输入")
        axes[0].axis('off')

        axes[1].imshow(sr_grid.permute(1, 2, 0))
        axes[1].set_title("超分辨率输出")
        axes[1].axis('off')

        axes[2].imshow(hr_grid.permute(1, 2, 0))
        axes[2].set_title("真实高分辨率")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "sample_grid.png"), dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试扩散模型超分辨率")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径，默认使用 final_model.pth")
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行测试
    config = Config()
    test_model(config, checkpoint_path=args.checkpoint)