"""
model.py - 扩散模型架构定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, time_emb_dim=None, cond_channels=None, dropout=0.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_channels = cond_channels

        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # 时间嵌入处理
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )

        # 条件处理（低分辨率图像）
        if cond_channels is not None:
            self.cond_conv = nn.Conv2d(cond_channels, out_channels, 1)

        # Dropout（防止过拟合，提高泛化能力）
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.conv1(x)

        # 添加时间嵌入
        if self.time_emb_dim is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]

        # 添加条件信息
        if self.cond_channels is not None and cond is not None:
            cond_resized = F.interpolate(cond, size=h.shape[2:], mode='bilinear', align_corners=False)
            cond_emb = self.cond_conv(cond_resized)
            h = h + cond_emb

        h = self.dropout_layer(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """注意力块"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        # q, k, v = rearrange(qkv, 'b (qkv c) h w -> qkv b c (h w)', qkv=3)
        ### 自己修改的
        def rearrange_qkv_native(qkv):
            batch_size, channels_times_3, height, width = qkv.shape
            channels = channels_times_3 // 3

            # 高效实现
            qkv_reshaped = qkv.view(batch_size, 3, channels, height * width)
            q = qkv_reshaped[:, 0, :, :]
            k = qkv_reshaped[:, 1, :, :]
            v = qkv_reshaped[:, 2, :, :]

            return q, k, v

        q, k, v = rearrange_qkv_native(qkv)

        # 计算注意力
        attn = torch.einsum('bci,bcj->bij', q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)

        return x + self.proj_out(out)


class UNet(nn.Module):
    """U-Net扩散模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 时间嵌入
        time_emb_dim = config.model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model_channels),
            nn.Linear(config.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 输入层
        self.input_conv = nn.Conv2d(config.channels, config.model_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = config.model_channels
        input_block_chans = [ch]
        for i, mult in enumerate(config.channel_mults):
            out_ch = config.model_channels * mult

            # 残差块
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, out_ch, time_emb_dim, config.channels,
                                  dropout=getattr(config, 'dropout', 0.0))
                )
                ch = out_ch
                input_block_chans.append(ch)

            # 下采样
            if i != len(config.channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)

        # 中间层
        _dp = getattr(config, 'dropout', 0.0)
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, config.channels, dropout=_dp)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, config.channels, dropout=_dp)

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(config.channel_mults))):
            out_ch = config.model_channels * mult

            # 上采样
            if i != len(config.channel_mults) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

            # 残差块
            for _ in range(config.num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                self.up_blocks.append(
                    ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, config.channels,
                                  dropout=getattr(config, 'dropout', 0.0))
                )
                ch = out_ch

        # 输出层
        self.output_norm = nn.GroupNorm(32, ch)
        self.output_conv = nn.Conv2d(ch, config.channels, 3, padding=1)

    def forward(self, x, time, cond):
        # 时间嵌入
        t = self.time_mlp(time)

        # 输入
        h = self.input_conv(x)

        # 保存跳跃连接
        skips = [h]

        # 下采样
        block_idx = 0
        sample_idx = 0
        for i in range(len(self.config.channel_mults)):
            for _ in range(self.config.num_res_blocks):
                h = self.down_blocks[block_idx](h, t, cond)
                skips.append(h)
                block_idx += 1
            if i != len(self.config.channel_mults) - 1:
                h = self.down_samples[sample_idx](h)
                skips.append(h)
                sample_idx += 1

        # 中间层
        h = self.mid_block1(h, t, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t, cond)

        # 上采样
        block_idx = 0
        sample_idx = 0
        for i in reversed(range(len(self.config.channel_mults))):
            if i != len(self.config.channel_mults) - 1:
                h = self.up_samples[sample_idx](h)
                sample_idx += 1
            for _ in range(self.config.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[block_idx](h, t, cond)
                block_idx += 1

        # 输出
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h


class DiffusionModel(nn.Module):
    """扩散模型包装器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 定义beta调度（注册为buffer以便随模型移动到GPU）
        if getattr(config, 'beta_schedule', 'linear') == 'cosine':
            betas = self._cosine_beta_schedule(config.timesteps)
        else:
            betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # 定义模型
        self.model = UNet(config)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """余弦beta调度，比线性调度产生更好的采样质量"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def forward(self, x, cond):
        """前向传播：预测噪声，同时返回预测的x0（供感知损失使用）"""
        # 随机采样时间步
        b = x.shape[0]
        t = torch.randint(0, self.config.timesteps, (b,), device=x.device).long()

        # 添加噪声
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        # 预测噪声
        predicted_noise = self.model(x_noisy, t, cond)

        # 预测 x0（用于感知损失/SSIM损失）
        x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / \
                  sqrt_alphas_cumprod_t.clamp(min=1e-6)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        return noise, predicted_noise, x0_pred

    @torch.no_grad()
    def sample(self, cond):
        """从条件生成样本"""
        device = next(self.model.parameters()).device
        batch_size = cond.shape[0]
        shape = (batch_size, self.config.channels,
                 self.config.hr_size, self.config.hr_size)

        # 从纯噪声开始
        x = torch.randn(shape, device=device)

        # 逐步去噪
        for i in reversed(range(self.config.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.model(x, t, cond)

            # 计算系数
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            beta = self.betas[t].view(-1, 1, 1, 1)

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 更新x
            x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x

    @torch.inference_mode()
    def sample_ddim(self, cond, ddim_steps=50, eta=0.0):
        """
        DDIM 加速采样，极致优化版：
        - 预计算所有 alpha 值为 GPU tensor，消除逐步 CPU→GPU 标量传输
        - eta=0 快速路径，跳过 sigma/noise 计算
        - 复用 t_tensor 避免重复分配
        - 25步 + cosine schedule 质量与50步接近
        """
        from torch.cuda.amp import autocast

        device = next(self.model.parameters()).device
        batch_size = cond.shape[0]
        shape = (batch_size, self.config.channels,
                 self.config.hr_size, self.config.hr_size)

        total_steps = self.config.timesteps

        # 均匀选取 ddim_steps 个时间步（从 T-1 到 0）
        step_ratio = total_steps // ddim_steps
        timesteps_seq = list(reversed(range(0, total_steps, step_ratio)))[:ddim_steps]

        # ★ 预计算所有步骤的 alpha 值为 GPU tensor，消除循环内 CPU 开销
        t_cur_list = timesteps_seq
        t_prev_list = timesteps_seq[1:] + [0]
        alpha_cur  = self.alphas_cumprod[torch.tensor(t_cur_list, device=device)]
        alpha_prev = self.alphas_cumprod[torch.tensor(t_prev_list, device=device)]
        # 预计算常用系数 [ddim_steps] → [ddim_steps, 1, 1, 1]
        sqrt_alpha_cur     = torch.sqrt(alpha_cur).view(-1, 1, 1, 1)
        sqrt_one_m_alpha   = torch.sqrt(1.0 - alpha_cur).view(-1, 1, 1, 1)
        sqrt_alpha_prev    = torch.sqrt(alpha_prev).view(-1, 1, 1, 1)
        sqrt_dir_coeff     = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)).view(-1, 1, 1, 1)

        # 预分配复用 tensor
        x = torch.randn(shape, device=device)
        cond = cond.to(device)
        t_tensor = torch.empty(batch_size, device=device, dtype=torch.long)

        with autocast():
            for step in range(ddim_steps):
                t_tensor.fill_(t_cur_list[step])

                # UNet 前向传播
                predicted_noise = self.model(x, t_tensor, cond)

                # 预测 x0
                x0_pred = (x - sqrt_one_m_alpha[step] * predicted_noise) / sqrt_alpha_cur[step]
                x0_pred.clamp_(-1.0, 1.0)

                if eta == 0:
                    # ★ 确定性采样快速路径：无 sigma、无 noise 分配
                    x = sqrt_alpha_prev[step] * x0_pred + sqrt_dir_coeff[step] * predicted_noise
                else:
                    alpha_cumprod_t = alpha_cur[step]
                    alpha_cumprod_t_prev = alpha_prev[step]
                    sigma = eta * torch.sqrt(
                        (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                        (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                    )
                    noise = torch.randn_like(x) if t_prev_list[step] > 0 else torch.zeros_like(x)
                    x = sqrt_alpha_prev[step] * x0_pred \
                        + torch.sqrt(torch.clamp(1 - alpha_cumprod_t_prev - sigma ** 2, min=0.0)) \
                        * predicted_noise + sigma * noise

        return x.float()