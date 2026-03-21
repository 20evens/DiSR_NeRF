"""
config.py - 超参数和配置设置
"""
import torch

class Config:
    # 数据配置
    data_root = "F:/Design/nerf-pytorch-master/data"  # 数据根目录
    train_dir = "nerf_synthetic/lego/train"  # 训练集目录
    val_dir = "nerf_synthetic/lego/val"  # 验证集目录
    test_dir = "nerf_synthetic/lego/test"  # 测试集目录

    # 图像配置
    hr_size = 256  # 高分辨率图像尺寸（2x SR：256×256）
    lr_size = 128  # 低分辨率图像尺寸
    scale_factor = 2  # 超分辨率倍率（2x：800→‖1600）
    channels = 3  # 图像通道数

    # 训练配置
    batch_size = 4  # 256分辨率显存占用适中
    num_workers = 4  # 多线程数据加载，消除I/O瓶颈
    learning_rate = 2e-4  # 较大学习率加速收敛
    num_epochs = 200  # 更多训练轮次提高质量
    save_interval = 10  # 保存模型的间隔周期数

    # 扩散模型配置
    timesteps = 1000  # 扩散步数
    beta_start = 0.0001  # 噪声调度起始值
    beta_end = 0.02  # 噪声调度结束值

    # 模型配置（2x SR版本，需删除旧checkpoint重新训练）
    model_channels = 128  # 模型基础通道数（hr=256时可恢复128）
    channel_mults = [1, 2, 4, 8]  # 通道倍增因子 → [128, 256, 512, 1024]
    num_res_blocks = 3  # 残差块数量（3→更深网络，更强表达能力）
    attention_resolutions = [32]  # 注意力分辨率（bottleneck=32×32）
    dropout = 0.1  # Dropout防止过拟合
    beta_schedule = 'cosine'  # 余弦调度比线性调度产生更好的采样质量

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径配置
    checkpoint_dir = "./checkpoints"
    result_dir = "./results"

    def __init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)