import torch.nn as nn
from .components import ResnetBlock, VBRCondition, FlexiblePrior, Downsample, Upsample, GDN1,quantize, NormalDistribution,MSFE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def gaussian_bpp(z, H, W, precision_bits=8, eps=1e-9):
    """
    计算[-1, 1]范围内连续特征的熵，保证结果 >= 0。
    
    参数:
    z: [B, C, H', W'] 输入特征，假设范围在 -1 到 1 之间
    H, W: 原图的高和宽 (用于归一化到像素点的bpp)
    precision_bits: 假设我们用多少位整数来表达这个范围。
                    8 表示相当于量化到 int8 (256个级别)
                    16 表示相当于 int16。
                    数值越大，保留的精度越高，算出来的 BPP 也会越高。
    """
    z=z.clamp(-1, 1)
    # 1. 确定“虚拟”量化步长 (Bin Size)
    # 数据范围是 2 (-1 到 1)，切分成 2^precision_bits 份
    # 例如 8-bit，步长约为 2/255 ≈ 0.0078
    total_levels = 2 ** precision_bits - 1
    bin_size = 2.0 / total_levels 
    
    # 2. 统计特征的分布参数 (均值和标准差)
    # 对于辅助信息，我们假设编码器会传输这些统计量，或者通过上下文预测
    mu = z.detach().mean(dim=(0, 2, 3), keepdim=True)
    var = z.detach().var(dim=(0, 2, 3), unbiased=False, keepdim=True) + eps
    std = var.sqrt()

    # 3. 定义标准高斯 CDF 函数
    def standardized_cumulative(x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    # 4. 计算概率质量 (Probability Mass)
    # 我们不看某一点的密度，而是看 z 落在 [z - bin/2, z + bin/2] 区间的概率面积
    # 中心化
    z_centered = z - mu
    
    # 将区间边界标准化
    # Upper bound CDF - Lower bound CDF
    # 这一步保证了 likelihood <= 1，所以 -log(likelihood) >= 0
    upper = standardized_cumulative((z_centered + 0.5 * bin_size) / std)
    lower = standardized_cumulative((z_centered - 0.5 * bin_size) / std)
    
    likelihood = (upper - lower).clamp(min=eps)

    # 5. 计算 Bits
    bits = -torch.log2(likelihood)
    
    # 6. 计算 BPP (Bits Per Pixel)
    B = z.shape[0]
    total_bits = bits.sum()
    bpp = total_bits / (B * H * W)

    return bpp

class w_compressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_extract = nn.Conv2d(3, 3, 3, padding=1)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        # 添加通道调整层
        self.adjust1 = nn.Conv2d(64, 128, 1)  # 调整f1池化后的通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )
        # 添加通道调整层
        self.adjust2 = nn.Conv2d(128, 192, 1)  # 调整f2池化后的通道数
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 192, 4, stride=2, padding=1),
            nn.GroupNorm(24, 192),
            nn.ReLU()
        )

    def forward(self, x):
        id = self.detail_extract(x) + x
        # id=x
        f1 = self.conv1(x)
        # 调整通道后相加
        f2 = self.conv2(f1) + self.adjust1(F.max_pool2d(f1, 2))
        f3 = self.conv3(f2) + self.adjust2(F.max_pool2d(f2, 2))
        return (id, f1, f2, f3)
    
class f_w_compressor(nn.Module):
    def __init__(self, detail_channels=16):
        super().__init__()
        # 细节增强层
        self.detail_extract = MSFE(in_planes=3, out_planes=3,map_reduce=1)
        self.f2_2=MSFE(in_planes=3,out_planes=64,map_reduce=1)
        self.f3_3=MSFE(in_planes=64,out_planes=128,map_reduce=8)
        self.f4_4=MSFE(in_planes=128,out_planes=192,map_reduce=8)
        # # 主路径
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, 4, 2, 1),
        #     nn.GroupNorm(8, 64),
        #     nn.ReLU(inplace=True)
        # )
        # self.adjust1 = nn.Conv2d(64, 128, 1)  # 64→128通道调整
        # # 修正通道调整层
        # self.channel_adjust1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(64, 64, 1),  # 输出通道与pooled_f1一致
        #     nn.Sigmoid()
        # )
        
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.GroupNorm(16, 128),
        #     nn.ReLU(inplace=True)
        # )
        
        # # 通道注意力
        # self.channel_attn = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(128, 128//16, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(128//16, 128, 1),
        #     nn.Sigmoid()
        # )
        # self.adjust2 = nn.Conv2d(128, 192, 1)  # 64→128通道调整
        
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 192, 4, 2, 1),
        #     nn.GroupNorm(24, 192),
        #     nn.ReLU(inplace=True)
        # )

    # def forward(self, x):
    #     identity = self.detail_extract(x)
    #     # f1 = self.conv1(x)
        
    #     # # 修正后的跨层连接
    #     # adjust_factor = self.channel_adjust1(f1)
    #     # pooled_f1 = F.max_pool2d(f1, 2) * adjust_factor  # 64通道 → 64通道
    #     f1=F.max_pool2d(self.f2_2(x), 2)
    #     f2=F.max_pool2d(self.f3_3(f1), 2)
    #     f3=F.max_pool2d(self.f4_4(f2), 2)
    #     # f2 = self.conv2(f1) + self.adjust1(pooled_f1)
    #     # f2 = f2 * self.channel_attn(f2)
        
    #     # f3 = self.conv3(f2) + self.adjust2(F.max_pool2d(f2, 2))
    #     return {"output":[identity, f1, f2, f3]}  # 返回字典，键为"output"
    def forward(self, x):
        B, _, H, W = x.shape

        identity = self.detail_extract(x)
        f1 = F.max_pool2d(self.f2_2(x), 2)
        f2 = F.max_pool2d(self.f3_3(f1), 2)
        f3 = F.max_pool2d(self.f4_4(f2), 2)

        # ===== 分析用 side-information BPP（不进 loss）=====
        with torch.no_grad():
            bpp_id = gaussian_bpp(identity, H, W)
            bpp_f1 = gaussian_bpp(f1, H, W)
            bpp_f2 = gaussian_bpp(f2, H, W)
            bpp_f3 = gaussian_bpp(f3, H, W)

            bpp_side = bpp_id + bpp_f1 + bpp_f2 + bpp_f3

        return {
            "output": [identity, f1, f2, f3],
            "bpp": bpp_side
        }


class big_w_compressor(nn.Module):
    """
    用于提取 y 图像的特征，包含完整的encoder-decoder结构，
    输出格式与compressor.forward()完全一致。
    """
    def __init__(self, dim=64, dim_mults=(1, 2, 3, 4), channels=3, latent_dim=None, vbr=False):
        super().__init__()
        self.vbr = vbr
        self.channels = channels
        self.dim = dim
        
        # Encoder部分（与compressor相同）
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        
        self.enc = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        # 最终潜在空间调整层
        final_dim = self.dims[-1]
        self.latent_dim = latent_dim or final_dim
        self.align_conv = nn.Conv2d(final_dim, self.latent_dim, 1)

        # Decoder部分（与compressor的decoder对称）
        self.reversed_dims = list(reversed([channels, *map(lambda m: dim * m, dim_mults)]))
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        
        self.dec = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        MSFE(in_planes=dim_out if not is_last else dim_in, 
                           out_planes=dim_out if not is_last else dim_in, 
                           map_reduce=1 if dim_out == 3 else 8),
                        VBRCondition(1, dim_out if not is_last else dim_in) if vbr else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

    def forward(self, y, cond=None):
        """
        输入: y 图像 (B, C, H, W)
        输出: 与compressor.forward()完全相同的字典格式
        """
        # Encoder路径（与compressor相同）
        B, _, H, W = y.shape
        current = y
        for resnet, vbrscaler, down in self.enc:
            current = resnet(current)
            if self.vbr:
                current = vbrscaler(current, cond)
            current = down(current)
        
        latent_y = self.align_conv(current)
        
        # Decoder路径生成多尺度特征（与compressor相同）
        output = self.decode(latent_y, cond)
        with torch.no_grad():
            bpp_side = 0.0
            for feat in output:
                bpp_side = bpp_side + gaussian_bpp(feat, H, W)
        
        return {
            "output": output,  # 多尺度特征列表，与compressor完全一致
            "bpp": bpp_side,      # 伪bpp
            "q_latent": latent_y,  # 对应x的q_latent
            "q_hyper_latent": torch.zeros_like(latent_y[:, :self.latent_dim//2]),  # 伪超先验
        }

    def decode(self, input, cond=None):
        """与compressor.decode()完全相同的解码过程"""
        output = []
        current = input
        
        for i, (resnet, msfm, vbrscaler, up) in enumerate(self.dec):
            current = resnet(current)
            current = msfm(current)  # 包含MSFE模块
            if self.vbr:
                current = vbrscaler(current, cond)
            current = up(current)
            output.append(current)
        
        return output[::-1]  # 反转顺序以匹配compressor的输出格式

    def get_latent(self, y, cond=None):
        """单独获取潜在特征（不经过decoder）"""
        current = y
        for resnet, vbrscaler, down in self.enc:
            current = resnet(current)
            if self.vbr:
                current = vbrscaler(current, cond)
            current = down(current)
        return self.align_conv(current)

    def get_features(self, y, cond=None):
        """获取多尺度特征（不包含伪bpp等压缩相关输出）"""
        latent = self.get_latent(y, cond)
        features = self.decode(latent, cond)
        return features

class compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = list(reversed([out_channels, *map(lambda m: dim * m, dim_mults)]))
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.vbr = vbr
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input, cond=None):
        for i, (resnet, vbrscaler, down) in enumerate(self.enc):
            input = resnet(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
        latent = input
        for i, (conv, vbrscaler, act) in enumerate(self.hyper_enc):
            input = conv(input)
            if self.vbr and i != (len(self.hyper_enc) - 1):
                input = vbrscaler(input, cond)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, vbrscaler, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            if self.vbr and i != (len(self.hyper_dec) - 1):
                input = vbrscaler(input, cond)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input, cond=None):
        output = []
        for i, (resnet, msfm,vbrscaler, down) in enumerate(self.dec):
            input = resnet(input)
            input=msfm(input)
            if self.vbr:
                input = vbrscaler(input, cond)
            input = down(input)
            output.append(input)
        return output[::-1]

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input, cond=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input, cond)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent, cond)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }


class Compressor(compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 3, 3, 3),
        hyper_dims_mults=(3, 3, 3),
        channels=3,
        out_channels=3,
        vbr=False,
    ):
        super().__init__(dim, dim_mults, hyper_dims_mults, channels, out_channels, vbr)
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        VBRCondition(1, dim_out) if self.vbr else nn.Identity(),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        MSFE(in_planes=dim_out if not is_last else dim_in,out_planes=dim_out if not is_last else dim_in ,map_reduce=1 if dim_out==3 else 8),
                        VBRCondition(1, dim_out if not is_last else dim_in)
                        if self.vbr
                        else nn.Identity(),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        VBRCondition(1, dim_out) if (self.vbr and not is_last) else nn.Identity(),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )


class Compressor22(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        mode="quant" #quant or pass
    ):
        super().__init__()
        self.channels = channels
        self.mode=mode
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input):
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input)
            input = down(input)
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
        hyper_latent = input
        if self.mode =="quant":
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        else:
            q_hyper_latent = quantize(hyper_latent, "bypass", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        if self.mode =="quant":
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        else:
            q_latent = quantize(latent, "bypass", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input):
        output = []
        for i, (resnet, msfm, up) in enumerate(self.dec):
            input = resnet(input)
            input = msfm(input)
            input = up(input)
            output.append(input)
        return output[::-1]

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            if self.mode =="quant":
                q_hyper_latent = quantize(hyper_latent, "noise")
                q_latent = quantize(latent, "noise")
            else:
                q_hyper_latent = quantize(hyper_latent, "bypass")
                q_latent = quantize(latent, "bypass")
        else:
            if self.mode =="quant":
                q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
                q_latent = quantize(latent, "dequantize", latent_distribution.mean)
            else:
                q_hyper_latent = quantize(hyper_latent, "bypass", self.prior.medians)
                q_latent = quantize(latent, "bypass", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input,bitrate_scale=None):
        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }


class ResnetCompressor(Compressor22):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        mode="quant"
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            mode
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            actual_out = dim_out if not is_last else dim_in
            self.dec.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, actual_out),
                    MSFE(in_planes=actual_out, out_planes=actual_out,
                        map_reduce=1 if actual_out == 3 else 8),
                    Upsample(actual_out, dim_out),
                ])
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )