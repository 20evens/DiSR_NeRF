"""
dataset.py - 数据集加载和处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import torchvision.transforms as transforms


class SRDataset(Dataset):
    """超分辨率数据集"""

    def __init__(self, data_dir, hr_size=3024, lr_size=378, scale_factor=8, is_train=True):
        self.data_dir = data_dir
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.scale_factor = scale_factor
        self.is_train = is_train

        # 获取所有图像文件
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + \
                           glob.glob(os.path.join(data_dir, "*.JPG")) + \
                           glob.glob(os.path.join(data_dir, "*.png")) + \
                           glob.glob(os.path.join(data_dir, "*.jpeg")) + \
                           glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(data_dir, "**", "*.JPG"), recursive=True) + \
                           glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True) + \
                           glob.glob(os.path.join(data_dir, "**", "*.jpeg"), recursive=True)
        self.image_paths = list(set(self.image_paths))  # 去重

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # 空间变换（输出仍为PIL图像）
        if is_train:
            self.spatial_transform = transforms.Compose([
                transforms.RandomCrop(hr_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.CenterCrop(hr_size),
            ])

        # 转为张量并归一化
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        hr_image = Image.open(img_path).convert('RGB')

        # 空间变换（裁剪、翻转），输出为PIL图像
        hr_patch = self.spatial_transform(hr_image)

        # 从HR裁剪下采样得到对应的LR图像
        lr_patch = hr_patch.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # 转为张量并归一化
        hr_tensor = self.to_tensor_norm(hr_patch)
        lr_tensor = self.to_tensor_norm(lr_patch)

        return hr_tensor, lr_tensor


def get_dataloaders(config, persistent_workers=False, prefetch_factor=2):
    """获取训练、验证和测试数据加载器"""
    train_dataset = SRDataset(
        os.path.join(config.data_root, config.train_dir),
        hr_size=config.hr_size,
        lr_size=config.lr_size,
        scale_factor=config.scale_factor,
        is_train=True
    )

    val_dataset = SRDataset(
        os.path.join(config.data_root, config.val_dir),
        hr_size=config.hr_size,
        lr_size=config.lr_size,
        scale_factor=config.scale_factor,
        is_train=False
    )

    test_dataset = SRDataset(
        os.path.join(config.data_root, config.test_dir),
        hr_size=config.hr_size,
        lr_size=config.lr_size,
        scale_factor=config.scale_factor,
        is_train=False
    )

    # DataLoader配置：persistent_workers保持worker进程，prefetch_factor预取数据
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers if config.num_workers > 0 else False,
        'prefetch_factor': prefetch_factor if config.num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader