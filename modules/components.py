import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from inspect import isfunction
from torch.autograd import Function

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm.auto import tqdm
import lpips
import time
from .components import *
import os
import PIL.Image as Image
from torch.autograd import Function
import sys
from pathlib import Path

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class CrossAlign(nn.Module):
    """
    线性注意力版本的 CrossAlign，显著降低显存占用
    """
    def __init__(self, dim_x, dim_w):
        super().__init__()
        self.query_conv = nn.Conv2d(dim_x, dim_w, 1)
        self.key_conv   = nn.Conv2d(dim_w, dim_w, 1)
        self.value_conv = nn.Conv2d(dim_w, dim_w, 1)
        self.scale = 1.0 / math.sqrt(dim_w)

    def forward(self, x_feat, w_feat):
        if w_feat.shape[2:] != x_feat.shape[2:]:
            w_feat = F.adaptive_avg_pool2d(w_feat, x_feat.shape[2:])

        Q = self.query_conv(x_feat)
        K = self.key_conv(w_feat)
        V = self.value_conv(w_feat)

        B, C, H, W = Q.shape
        Q = Q.flatten(2)  # [B, C, HW]
        K = K.flatten(2)
        V = V.flatten(2)

        # 使用 softmax 近似线性化注意力
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        # 先算分母：Kᵀ·1
        KV = torch.bmm(K, V.transpose(1, 2))       # [B, C, C]
        Z = 1 / (torch.bmm(Q.transpose(1, 2), K.sum(dim=2, keepdim=True)) + 1e-6)

        out = torch.bmm(KV, Q) * Z.transpose(1, 2)  # [B, C, HW]
        out = out.view(B, C, H, W)
        return out
    

    
def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')

    # Save img_recon
    img_recon_path = os.path.join(path, 'img_recon')
    if not os.path.exists(img_recon_path):
        os.makedirs(img_recon_path)
    img_recon_final = Image.fromarray(img_recon, 'RGB')
    img_recon_final.save(os.path.join(img_recon_path, name + '.png'))

    # Save img
    img_path = os.path.join(path, 'img')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img_final = Image.fromarray(img, 'RGB')
    img_final.save(os.path.join(img_path, name + '.png'))

def sample_latent_quant_noise(w_tensor, bpp, Q=64, alpha=1.0):
    B, C, H, W = w_tensor.shape
    device = w_tensor.device
   
    # 1. 通道尺度 (保持不变)
    chan_std = w_tensor.detach().var(dim=(2,3), unbiased=False).sqrt() + 1e-8
    chan_std = chan_std.view(B, C, 1, 1)
    
    # 2. bpp → delta (保持不变)
    if not isinstance(bpp, torch.Tensor):
        bpp = torch.full((B,), float(bpp), device=device)
    else:
        bpp = bpp.float().view(B)
    delta = torch.exp(-6.0 * bpp).view(B,1,1,1).clamp(1e-3, 1.0)
    
    # ================= 修改核心区域 =================
    
    # A. 生成量化噪声 (均匀分布)
    # 范围大约在 [-0.5, 0.5] * (2/Q) 左右，或者 [-1, 1] 取决于你的 Q 归一化
    # 你的原代码: randint(-Q, Q+1) / Q  --> 范围是 [-1, 1]
    discrete = torch.randint(-Q, Q+1, (B,C,H,W), device=device, dtype=torch.float32) / Q
    
    # B. 生成高斯噪声 (标准正态分布)
    # 范围是无界的，但主要集中在 [-2, 2]
    gaussian = torch.randn_like(discrete)
    
    # C. 混合策略 (关键！)
    # 方案 1: 激进混合 (强高斯，保留一定量化特征)
    # 0.7 的高斯保证了扩散生成的动力
    # 0.3 的量化保证了模型见过这种伪影
    # noise = 0.7 * gaussian + 0.3 * discrete
    
    # 方案 2: 如果你希望更像标准扩散，高斯比例要更高
    noise = 1 * gaussian 
    return noise
    # ==============================================
    
    # # 3. 缩放 (保持不变)
    # # 注意：因为现在 noise 的方差变大了 (主要是高斯变大了)，
    # # 这里的 alpha 可能需要适当调小一点点，或者保持 1.0 让模型适应强噪声
    # noise = noise * (delta * chan_std) * alpha
    
    # # 4. 截断
    # # 因为高斯是无界的，混合后可能会有大值，建议稍微放宽截断
    # return noise.clamp(-3, 3)


class SpatialGatedFusionLayer(nn.Module):
    """
    适用于立体图像/有视差场景的融合层。
    替代 AdaIN，使用 SFT (Spatial Feature Transform) 思想。
    
    特点：
    1. 空间敏感：不压缩空间维度，保留 w 中的纹理和边缘细节。
    2. 自动对齐感知：通过拼接 (x, w) 让网络判断视差和遮挡。
    3. 门控机制：使用 Sigmoid 生成信任图 (Confidence Map)。
    4. 零初始化：保证初始状态为 Identity，训练极其稳定。
    """
    def __init__(self, content_channels, style_channels):
        super().__init__()

        # 输入通道是 x 和 w 的总和，这样卷积层能同时看到两者
        in_channels = content_channels + style_channels
        
        # --- 分支 1：特征生成网络 (生成要补充的残差信息) ---
        self.residual_net = nn.Sequential(
            # 第一层卷积融合 x 和 w 的信息
            nn.Conv2d(in_channels, content_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第二层卷积生成最终的修正量
            nn.Conv2d(content_channels, content_channels, kernel_size=3, padding=1)
        )

        # --- 分支 2：门控网络 (生成像素级的融合强度) ---
        # 输出通道为 content_channels，表示每个通道、每个像素都有独立的权重
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels, content_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出限制在 0~1 之间
        )

        # --- 稳定性初始化 ---
        # 将 residual_net 最后一层权重初始化为 0
        # 效果：初始时刻 delta = 0，output = content_feature
        # 这比手动乘 0.5 更稳健，能防止 w 中的噪音在训练初期破坏 x
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, content_feature, style_feature):
        """
        content_feature (x): 当前解码的特征 (主视角)
        style_feature (w): 边信息特征 (辅助视角，有视差)
        """
        # 1. 拼接：让网络同时看到 x 和 w，以此判断对齐情况
        # 假设 content 和 style 的 H, W 尺寸一致
        cat_feat = torch.cat([content_feature, style_feature], dim=1)
        
        # 2. 计算修正量 (Delta)
        # 网络会学习从 w 中提取 x 缺失的纹理细节
        delta = self.residual_net(cat_feat)
        
        # 3. 计算门控 (Gate)
        # 0 表示 w 该处不可信(遮挡/视差过大)，1 表示该处可以融合
        gate = self.gate_net(cat_feat)
        
        # 4. 融合：原始特征 + 门控 * 修正量
        output = content_feature + gate * delta
        
        return output


# ----------------------------------------------------
# 管理器模块：多尺度空间融合 (Multi-Scale Fusion)
# ----------------------------------------------------
class MultiScaleSpatialFusion(nn.Module):
    """
    管理多个尺度的空间门控融合。
    可以直接替换原本的 MultiScaleStableAdaINFusion。
    """
    def __init__(self, z_x_channels_list, w_channels_list):
        super().__init__()
        
        assert len(z_x_channels_list) == len(w_channels_list), \
            "z_x 和 w 的尺度数量必须相同"
            
        self.fusion_layers = nn.ModuleList()
        for z_ch, w_ch in zip(z_x_channels_list, w_channels_list):
            # 使用新的空间门控融合层
            self.fusion_layers.append(SpatialGatedFusionLayer(z_ch, w_ch))

    def forward(self, z_x_features, w_features):
        """
        z_x_features: 列表，包含不同尺度的 x 特征
        w_features:   列表，包含不同尺度的 w 特征
        """
        fused_features = []
        # 遍历每一层进行融合
        for i, layer in enumerate(self.fusion_layers):
            fused_i = layer(z_x_features[i], w_features[i])
            fused_features.append(fused_i)
            
        return fused_features
# ----------------------------------------------------
# 核心模块：GPT 推荐的、更稳定的“残差 AdaIN”层
# ----------------------------------------------------
class StableAdaINLayer(nn.Module):
    """
    一个更稳定的 AdaIN 版本，适用于条件生成和信息融合任务。
    特点：
    1. 使用残差连接，保留原始 content_feature 的信息。
    2. 使用 tanh 限制 gamma 的范围，防止梯度爆炸。
    3. 融合强度 strength 是一个可学习的参数。
    """
    def __init__(self, content_channels, style_channels):
        super().__init__()

        # 从 style_feature (w) 生成 gamma 和 beta 的网络
        self.gamma_generator = nn.Conv2d(style_channels, content_channels, kernel_size=1)
        self.beta_generator = nn.Conv2d(style_channels, content_channels, kernel_size=1)

        # 可学习的融合强度，初始化为 1.0
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, content_feature, style_feature):
        # --- 1. 对内容特征 (z_x) 进行实例归一化 ---
        mu = torch.mean(content_feature, dim=[2, 3], keepdim=True)
        sigma = torch.std(content_feature, dim=[2, 3], keepdim=True)
        normalized_content = (content_feature - mu) / (sigma + 1e-5)

        # --- 2. 从风格特征 (w) 生成受限的 gamma 和 beta ---
        # 使用 tanh 将 gamma 限制在 (-1, 1) 之间，增强稳定性
        gamma = torch.tanh(self.gamma_generator(style_feature))
        # beta (偏移) 通常不需要限制
        beta = self.beta_generator(style_feature)

        # --- 3. 计算 AdaIN 调制后的特征 ---
        adain_modulation = gamma * normalized_content + beta
        
        # --- 4. 核心改进：使用残差连接进行融合 ---
        # 将 AdaIN 作为对原始 content_feature 的一个修正量
        # 乘以 0.5 是一个额外的稳定化技巧，防止初始修正过大
        output = content_feature + self.strength * adain_modulation * 0.5

        return output

# ----------------------------------------------------
# 管理器模块：使用新的 StableAdaINLayer
# ----------------------------------------------------
class MultiScaleStableAdaINFusion(nn.Module):
    """
    管理多个尺度的、更稳定的 AdaIN 融合。
    """
    def __init__(self, z_x_channels_list, w_channels_list):
        super().__init__()
        
        assert len(z_x_channels_list) == len(w_channels_list), \
            "z_x 和 w 的尺度数量必须相同"
            
        self.fusion_layers = nn.ModuleList()
        for z_channels, w_channels in zip(z_x_channels_list, w_channels_list):
            # 使用我们新的、更稳定的 AdaIN 层
            self.fusion_layers.append(StableAdaINLayer(z_channels, w_channels))

    def forward(self, z_x_features, w_features):
        fused_features = []
        for i in range(len(z_x_features)):
            fused_i = self.fusion_layers[i](z_x_features[i], w_features[i])
            fused_features.append(fused_i)
            
        return fused_features
    
class MSFE(nn.Module):
    """三分支，渐进膨胀率 + 通道注意力 + 可学习门控"""
    def __init__(self, in_planes, out_planes, stride=1, map_reduce=8):
        super(MSFE, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        # Branch 0: 局部特征 (不变)
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=1, relu=False)
        )
        # Branch 1: dilation 5→3
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3,
                      kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes,
                      kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3, relu=False)
        )
        # Branch 2: dilation 保持5→改7
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3,
                      kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes,
                      kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
                      stride=1, padding=7, dilation=7, relu=False)
        )

        # 融合 (和原版一样是 6*inter_planes → out)
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes,
                                    kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
                                  stride=stride, relu=False)

        # [新增] 通道注意力 (两个1×1 conv, 额外参数≈0)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, max(out_planes // 4, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_planes // 4, 1), out_planes, 1, bias=False),
            nn.Sigmoid()
        )

        # [新增] 可学习门控 (1个参数/通道，替代固定scale=0.1)
        self.gate = nn.Parameter(torch.full((1, out_planes, 1, 1), 0.1))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)

        # 通道注意力
        out = out * self.channel_attn(out)

        # 可学习门控残差
        short = self.shortcut(x)
        out = torch.sigmoid(self.gate) * out + short

        return self.relu(out)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)  # Swish 公式：x * σ(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None, dropout=0.):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv_x = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_qkv_x = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        # self.to_qkv_y = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_qkv_y = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        qkv_x = self.to_qkv_x(x).chunk(3, dim=-1)
        qkv_y = self.to_qkv_y(y).chunk(3, dim=-1)
        q_x, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_x)
        _, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_y)

        dots = torch.matmul(q_x, k_y.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v_y)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, input_size, num_filters=192, heads=1, ch_patch_size=1, num_patches=4, dim=None, dim_head=None,
                 dropout=0.):
        super().__init__()

        assert num_filters % ch_patch_size == 0, 'num_filters must be divisible by the patch size.'
        self.patch_size = [None] * 3
        self.patch_size[0] = ch_patch_size
        self.patch_size[1] = input_size[0] // num_patches
        self.patch_size[2] = input_size[1] // num_patches
        patch_dim = self.patch_size[1] * self.patch_size[2] * ch_patch_size
        self.num_patches=num_patches
        self.ch_patch_size=ch_patch_size
        # # 调试输出
        # print(f"input_size: {input_size}")
        # print(f"num_patches: {num_patches}")
        # print(f"patch_size[1]: {self.patch_size[1]} (计算为 {input_size[0]} // {num_patches})")
        # print(f"patch_size[2]: {self.patch_size[2]} (计算为 {input_size[1]} // {num_patches})")
        if dim is None:
            dim = patch_dim

        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)', p0=ch_patch_size,
                      p1=self.patch_size[1], p2=self.patch_size[2]),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_y = nn.Sequential(
            Rearrange('b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)', p0=ch_patch_size,
                      p1=self.patch_size[1], p2=self.patch_size[2]),
            nn.Linear(patch_dim, dim),
        )
        self.unpack_embedding_y = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (c h w) (p0 p1 p2) -> b (c p0) (h p1) (w p2)', h=num_patches, w=num_patches,
                      p0=ch_patch_size, p1=self.patch_size[1],
                      p2=self.patch_size[2]),
        )

        self.norm = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim, heads, dim_head, dropout)
        self.to_out = nn.Sequential(nn.Conv2d(num_filters*2, num_filters, 1))

    def forward(self, x, y):
        # print(f"Input shape: x={x.shape}, y={y.shape}")  # Debug
        # print(self.num_patches)
        # print(self.ch_patch_size)
        x_emb = self.to_patch_embedding_x(x)
        y_emb = self.to_patch_embedding_y(y)
        x_norm = self.norm(x_emb)
        y_norm = self.norm(y_emb)

        y = self.attn(x_norm, y_norm)
        y = self.unpack_embedding_y(y)

        aa = torch.cat((x, y), 1)

        return self.to_out(aa)+x



class PreNorm_context(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm(x)
        return self.fn(x, context) if context is not None else self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)
# class Upsample(nn.Module):
#     def __init__(self, dim_in, dim_out=None):
#         super().__init__()
#         if dim_out is None:
#             dim_out = dim_in
        
#         # 我们将使用 'nearest' 插值模式进行上采样，然后接一个标准的卷积层
#         # 'nearest' 模式速度快，且能有效避免引入新的伪影
#         # 卷积核大小为 3，padding 为 1，可以在不改变分辨率的情况下进行特征变换
#         self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

#     def forward(self, x):
#         # 1. 先使用插值将特征图的分辨率放大两倍
#         #    align_corners=False 是推荐的设置
#         x = F.interpolate(x, scale_factor=2, mode='nearest')
        
#         # 2. 然后通过一个标准卷积层来学习和细化上采样后的特征
#         x = self.conv(x)
        
#         return x

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, large=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 7 if large else 3, padding=3 if large else 1), LayerNorm(dim_out), nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, large)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(time_emb):
            # h = h + self.mlp(time_emb)[:, :, None, None]
            h = h + self.mlp(time_emb).view(-1, h.size(1), 1, 1)

        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    """标准 Multi-Head Self-Attention，用于 UNet mid block"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h d i", attn, v)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, n_layer=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.cur_states = [None for i in range(n_layer)]
        self.n_layer = n_layer

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.input_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
            ]
            + [
                nn.Conv2d(
                    in_channels=self.hidden_dim + self.hidden_dim,
                    out_channels=4 * self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                )
                for i in range(n_layer - 1)
            ]
        )

    def step_forward(self, input_tensor, layer_index=0):
        assert self.cur_states[layer_index] is not None
        h_cur, c_cur = self.cur_states[layer_index]
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.convs[layer_index](combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        self.cur_states[layer_index] = (h_next, c_next)

        return h_next

    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor

    def init_hidden(self, batch_shape):
        B, _, H, W = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = (
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device,),
                torch.zeros(B, self.hidden_dim, H, W, device=self.convs[0].weight.device,),
            )


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,  # for candidate neural memory
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=self.conv_cans[0].weight.device)

    def step_forward(self, input_tensor, index):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next
    
    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor


class VBRCondition(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.scale = nn.Conv2d(input_dim, output_dim, 1)
        self.shift = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, input, cond):
        cond = cond.reshape(-1, 1, 1, 1)
        scale = self.scale(cond)
        shift = self.shift(cond)
        return input * scale + shift


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=.1, reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GDN1(GDN):
    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(torch.abs(inputs), gamma, beta)
        # norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class PriorFunction(nn.Module):
    #  A Custom Function described in Balle et al 2018. https://arxiv.org/pdf/1802.01436.pdf
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, parallel_dims, in_features, out_features, scale, bias=True):
        super(PriorFunction, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(parallel_dims, 1, 1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.5, 0.5)

    def forward(self, input, detach=False):
        # input shape (channel, batch_size, in_features)
        if detach:
            return torch.matmul(input, F.softplus(self.weight.detach())) + self.bias.detach()
        return torch.matmul(input, F.softplus(self.weight)) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)


class FlexiblePrior(nn.Module):
    '''
        A prior model described in Balle et al 2018 Appendix 6.1 https://arxiv.org/pdf/1802.01436.pdf
        return the boxshape likelihood
    '''
    def __init__(self, channels=256, dims=[3, 3, 3], init_scale=10.):
        super(FlexiblePrior, self).__init__()
        dims = [1] + dims + [1]
        self.chain_len = len(dims) - 1
        scale = init_scale**(1 / self.chain_len)
        h_b = []
        for i in range(self.chain_len):
            init = np.log(np.expm1(1 / scale / dims[i + 1]))
            h_b.append(PriorFunction(channels, dims[i], dims[i + 1], init))
        self.affine = nn.ModuleList(h_b)
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels, 1, 1, 1, dims[i + 1])) for i in range(self.chain_len - 1)])

        # optimize the medians to fix the offset issue
        self._medians = nn.Parameter(torch.zeros(1, channels, 1, 1))
        # self.register_buffer('_medians', torch.zeros(1, channels, 1, 1))

    @property
    def medians(self):
        return self._medians.detach()

    def cdf(self, x, logits=True, detach=False):
        x = x.transpose(0, 1).unsqueeze(-1)  # C, N, H, W, 1
        if detach:
            for i in range(self.chain_len - 1):
                x = self.affine[i](x, detach)
                x = x + torch.tanh(self.a[i].detach()) * torch.tanh(x)
            if logits:
                return self.affine[-1](x, detach).squeeze(-1).transpose(0, 1)
            return torch.sigmoid(self.affine[-1](x, detach)).squeeze(-1).transpose(0, 1)

        # not detached
        for i in range(self.chain_len - 1):
            x = self.affine[i](x)
            x = x + torch.tanh(self.a[i]) * torch.tanh(x)
        if logits:
            return self.affine[-1](x).squeeze(-1).transpose(0, 1)
        return torch.sigmoid(self.affine[-1](x)).squeeze(-1).transpose(0, 1)

    def pdf(self, x):
        cdf = self.cdf(x, False)
        jac = torch.ones_like(cdf)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=jac)[0]
        return pdf

    def get_extraloss(self):
        target = 0
        logits = self.cdf(self._medians, detach=True)
        extra_loss = torch.abs(logits - target).sum()
        return extra_loss

    def likelihood(self, x, min=1e-9):
        lower = self.cdf(x - 0.5, True)
        upper = self.cdf(x + 0.5, True)
        sign = -torch.sign(lower + upper).detach()
        upper = torch.sigmoid(upper * sign)
        lower = torch.sigmoid(lower * sign)
        return LowerBound.apply(torch.abs(upper - lower), min)

    def icdf(self, xi, method='bisection', max_iterations=1000, tol=1e-9, **kwargs):
        if method == 'bisection':
            init_interval = [-1, 1]
            left_endpoints = torch.ones_like(xi) * init_interval[0]
            right_endpoints = torch.ones_like(xi) * init_interval[1]

            def f(z):
                return self.cdf(z, logits=False, detach=True) - xi

            while True:
                if (f(left_endpoints) < 0).all():
                    break
                else:
                    left_endpoints = left_endpoints * 2
            while True:
                if (f(right_endpoints) > 0).all():
                    break
                else:
                    right_endpoints = right_endpoints * 2

            for i in range(max_iterations):
                mid_pts = 0.5 * (left_endpoints + right_endpoints)
                mid_vals = f(mid_pts)
                pos = mid_vals > 0
                non_pos = torch.logical_not(pos)
                neg = mid_vals < 0
                non_neg = torch.logical_not(neg)
                left_endpoints = left_endpoints * non_neg.float() + mid_pts * neg.float()
                right_endpoints = right_endpoints * non_pos.float() + mid_pts * pos.float()
                if (torch.logical_and(non_pos, non_neg)).all() or torch.min(right_endpoints - left_endpoints) <= tol:
                    print(f'bisection terminated after {i} its')
                    break

            return mid_pts
        else:
            raise NotImplementedError

    def sample(self, img, shape):
        uni = torch.rand(shape, device=img.device)
        return self.icdf(uni)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_tensor(a, t, place_holder=None):
    return a[t, torch.arange(len(t))]


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# def cosine_beta_schedule(timesteps, s = 0.008):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
#     alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def noise(input, scale):
    return input + scale*(torch.rand_like(input) - 0.5)


def round_w_offset(input, loc):
    diff = STERound.apply(input - loc)
    return diff + loc


def quantize(x, mode='noise', offset=None):
    if mode == 'noise':
        return noise(x, 1)
    elif mode == 'round':
        return STERound.apply(x)
    elif mode == 'dequantize':
        return round_w_offset(x, offset)
    elif mode == 'bypass':
        # 旁路：不量化，直接透传
        return x
    else:
        raise NotImplementedError(f"Unsupported quantize mode: {mode}")


class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class UpperBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs <= b
        pass_through_2 = grad_output > 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class NormalDistribution:
    '''
        A normal distribution
    '''
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc.detach()

    def std_cdf(self, inputs):
        half = 0.5
        const = -(2**-0.5)
        return half * torch.erfc(const * inputs)

    def sample(self):
        return self.scale * torch.randn_like(self.scale) + self.loc

    def likelihood(self, x, min=1e-9):
        x = torch.abs(x - self.loc)
        upper = self.std_cdf((.5 - x) / self.scale)
        lower = self.std_cdf((-.5 - x) / self.scale)
        return LowerBound.apply(upper - lower, min)

    def scaled_likelihood(self, x, s=1, min=1e-9):
        x = torch.abs(x - self.loc)
        s = s * .5
        upper = self.std_cdf((s - x) / self.scale)
        lower = self.std_cdf((-s - x) / self.scale)
        return LowerBound.apply(upper - lower, min)