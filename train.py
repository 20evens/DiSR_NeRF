"""
train.py - 训练扩散模型
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 Windows 上 OpenMP 库冲突

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import numpy as np

import torchvision.models as tv_models
import torch.nn.functional as F

from config import Config
from model import DiffusionModel
from dataset import get_dataloaders
from utils import save_checkpoint, load_checkpoint, calculate_psnr, calculate_ssim

# GPU 极限优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PerceptualLoss(nn.Module):
    """
    VGG16感知损失：强制模型生成感知自然的图像细节，大幅减少噪点和模糊。
    对x0_pred和hr_images计算VGG特征图的L1距离。
    """
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features[:16]
        vgg = vgg.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    def forward(self, pred, target):
        pred_01   = pred   * 0.5 + 0.5
        target_01 = target * 0.5 + 0.5
        pred_n   = (pred_01   - self.mean) / self.std
        target_n = (target_01 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


def ssim_loss(pred, target, window_size=11):
    """
    1 - SSIM 损失（简化实现）。
    测量pred和target的结构相似性，防止模型生成结构揭失。
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2
    mu_p  = F.avg_pool2d(pred,   window_size, stride=1, padding=pad)
    mu_t  = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu_p2, mu_t2 = mu_p ** 2, mu_t ** 2
    mu_pt = mu_p * mu_t
    sig_p  = F.avg_pool2d(pred   ** 2, window_size, stride=1, padding=pad) - mu_p2
    sig_t  = F.avg_pool2d(target ** 2, window_size, stride=1, padding=pad) - mu_t2
    sig_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2))
    return 1.0 - ssim_map.mean()


def train_epoch(model, dataloader, optimizer, criterion, epoch, config, writer,
               scaler=None, perceptual_loss_fn=None):
    """
    训练一个epoch：组合MSE噪声损失 + VGG感知损失 + SSIM损失。
    感知损失和SSIM损失作用于预测的x0，改善输出图像的感知质量和道亚管维度。
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (hr_images, lr_images) in enumerate(progress_bar):
        hr_images = hr_images.to(config.device, non_blocking=True)
        lr_images = lr_images.to(config.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            noise, predicted_noise, x0_pred = model(hr_images, lr_images)

            # 基础 MSE 噪声损失
            loss_mse = criterion(predicted_noise, noise)

            # VGG 感知损失（作用于x0_pred vs hr）
            loss_percep = torch.tensor(0.0, device=config.device)
            if perceptual_loss_fn is not None:
                loss_percep = perceptual_loss_fn(x0_pred, hr_images)

            # SSIM 损失（测量结构相似性）
            loss_ssim = ssim_loss(x0_pred, hr_images)

            # 加权组合：MSE为主，感知+SSIM为辅
            loss = loss_mse + 0.1 * loss_percep + 0.05 * loss_ssim

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            mse=f"{loss_mse.item():.4f}",
            perc=f"{loss_percep.item():.4f}",
            ssim=f"{loss_ssim.item():.4f}"
        )

        if writer is not None and batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss',          loss.item(),        step)
            writer.add_scalar('Train/Loss_MSE',      loss_mse.item(),    step)
            writer.add_scalar('Train/Loss_Perceptual', loss_percep.item(), step)
            writer.add_scalar('Train/Loss_SSIM',     loss_ssim.item(),   step)

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, config):
    """验证模型（使用DDIM加速采样）"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for hr_images, lr_images in tqdm(dataloader, desc="Validation"):
            # 移动到设备
            hr_images = hr_images.to(config.device)
            lr_images = lr_images.to(config.device)

            # 前向传播
            noise, predicted_noise, _ = model(hr_images, lr_images)

            # 计算损失
            loss = criterion(predicted_noise, noise)
            total_loss += loss.item()

            # 生成样本并计算指标（使用DDIM 50步加速，速度×20）
            for i in range(min(2, hr_images.shape[0])):  # 减少到2个样本
                sample = model.sample_ddim(lr_images[i:i + 1], ddim_steps=50)

                # 反归一化
                hr_np = (hr_images[i].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0)
                sample_np = (sample[0].cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0)

                # 计算PSNR和SSIM
                psnr = calculate_psnr(hr_np, sample_np)
                ssim = calculate_ssim(hr_np, sample_np)

                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0

    return avg_loss, avg_psnr, avg_ssim


def main():
    # 加载配置
    config = Config()

    # 创建tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(config.result_dir, 'logs'))

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(config, persistent_workers=True, prefetch_factor=2)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 初始化模型
    model = DiffusionModel(config).to(config.device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # VGG 感知损失
    try:
        perceptual_loss_fn = PerceptualLoss(config.device)
        print("感知损失 (VGG16): 已启用")
    except Exception as e:
        perceptual_loss_fn = None
        print(f"感知损失不可用，仅使用MSE: {e}")

    # 混合精度训练（FP16，速度×2，显存÷2）
    scaler = GradScaler()

    # 余弦学习率调度器（平滑衰减，提高最终模型质量）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    # 加载检查点（如果存在）
    start_epoch = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pth")
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, checkpoint_path, config.device
        )
        print(f"从检查点恢复，起始周期: {start_epoch}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"混合精度训练: 已启用 (FP16)")
    print(f"优化器: AdamW (weight_decay=1e-4)")
    print(f"学习率调度: CosineAnnealing ({config.learning_rate} → 1e-6)")
    print(f"Beta调度: {getattr(config, 'beta_schedule', 'linear')}")
    print(f"Dropout: {getattr(config, 'dropout', 0.0)}")

    # 训练循环
    best_psnr = 0
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'=' * 50}")

        # 训练（混合精度 + 感知损失）
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            epoch, config, writer, scaler=scaler,
            perceptual_loss_fn=perceptual_loss_fn
        )
        scheduler.step()  # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"训练损失: {train_loss:.6f}  lr: {current_lr:.2e}")

        # 验证（每5个epoch验证一次，节省时间）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, config)
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证PSNR: {val_psnr:.4f} dB")
            print(f"验证SSIM: {val_ssim:.4f}")

            # 记录到tensorboard
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/PSNR', val_psnr, epoch)
            writer.add_scalar('Validation/SSIM', val_ssim, epoch)
        else:
            val_psnr = 0  # 非验证epoch，不更新best_psnr

        # 保存最新检查点（用于恢复训练）
        save_checkpoint(
            model, optimizer, epoch + 1,
            os.path.join(config.checkpoint_dir, "latest.pth")
        )

        # 保存定期检查点
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            )

        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(config.checkpoint_dir, "best_model.pth")
            )
            print(f"保存最佳模型，PSNR: {best_psnr:.4f} dB")

    # 保存最终模型
    save_checkpoint(
        model, optimizer, config.num_epochs,
        os.path.join(config.checkpoint_dir, "final_model.pth")
    )

    writer.close()
    print("训练完成！")


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    np.random.seed(42)
    main()