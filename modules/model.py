import warnings
warnings.filterwarnings("ignore")
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

current_path = Path(__file__).resolve()
parent_dir_a_path = current_path.parent
parent_dir_path = parent_dir_a_path.parent
sys.path.append(str(parent_dir_path))

import config


class CrossAlign(nn.Module):
    """线性注意力版本的 CrossAlign，显著降低显存占用"""
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
        Q = Q.flatten(2)
        K = K.flatten(2)
        V = V.flatten(2)

        Q = F.elu(Q) + 1
        K = F.elu(K) + 1

        KV = torch.bmm(K, V.transpose(1, 2))
        Z = 1 / (torch.bmm(Q.transpose(1, 2), K.sum(dim=2, keepdim=True)) + 1e-6)

        out = torch.bmm(KV, Q) * Z.transpose(1, 2)
        out = out.view(B, C, H, W)
        return out


class PrivateYEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 64, 128, 192]
        self.blocks = nn.ModuleList()

        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else 3
            out_ch = channels[i]
            downsample = (i > 0)
            self.blocks.append(PrivateBlock(in_ch, out_ch, downsample=downsample))

    def forward(self, image_y):
        z_y = image_y
        feats = []
        for block in self.blocks:
            z_y = block(z_y)
            feats.append(z_y)
        return feats


class PrivateBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 5, stride=stride, padding=2)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch)

        if downsample:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=2),
                nn.BatchNorm2d(out_ch)
            ) if in_ch != out_ch else nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat = (feat3 + feat5) / 2
        feat = self.act(self.norm(feat))
        feat = self.se(feat)
        residual = self.residual(x)
        out = feat + residual
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')

    img_recon_path = os.path.join(path, 'img_recon')
    if not os.path.exists(img_recon_path):
        os.makedirs(img_recon_path)
    Image.fromarray(img_recon, 'RGB').save(os.path.join(img_recon_path, name + '.png'))

    img_path = os.path.join(path, 'img')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    Image.fromarray(img, 'RGB').save(os.path.join(img_path, name + '.png'))


class DICM(nn.Module):
    def __init__(
            self,
            device,
            denoise_fn,
            context_fn,
            context_w_p,
            context_w,
            channels=3,
            num_timesteps=1000,
            loss_type="l1",
            clip_noise="half",
            vbr=False,
            lagrangian=1e-3,
            pred_mode="noise",
            var_schedule="linear",
            aux_loss_weight=0,
            aux_loss_type="l1",
            unroll_steps=1,
            detach_unroll=False,
            rate_warmup_epochs=10,
            aid_ablation=False,
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.context_w_p = context_w_p
        self.encode_w = context_w
        self.clip_noise = clip_noise
        self.vbr = vbr
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian          # 当前生效的 β（可被 trainer 动态更新）
        self.var_schedule = var_schedule
        self.sample_steps = num_timesteps
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_type = aux_loss_type
        self.device = device
        self.loss_weight_min = 5.0
        self.ee = False

        self.unroll_steps = unroll_steps
        self.detach_unroll = detach_unroll
        self.rate_warmup_epochs = rate_warmup_epochs

        # AID 消融开关只控制辅助解耦目标的损失权重；默认关闭时保持原训练行为。
        # 即使开启消融，下面的辅助分支仍保留在前向图中并乘以 0，
        # 这样不会改变模块/参数集合，也不会在 DDP 的静态参数同步中引入未使用参数。
        self.aid_ablation = bool(aid_ablation)

        assert pred_mode in ["noise", "x", "renoise"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)

        if aux_loss_weight > 0:
            self.loss_fn_vgg = lpips.LPIPS(net="vgg", eval_mode=False, verbose=False).to(self.device)
        else:
            self.loss_fn_vgg = None

        if var_schedule == "cosine":
            train_betas = cosine_beta_schedule(num_timesteps)
        elif var_schedule == "linear":
            train_betas = linear_beta_schedule(num_timesteps)

        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)
        (num_timesteps,) = train_betas.shape
        self.num_timesteps = int(num_timesteps)

        self.register_buffer("train_betas", to_torch(train_betas).to(self.device))
        self.register_buffer("train_alphas_cumprod", to_torch(train_alphas_cumprod).to(self.device))
        self.register_buffer("train_sqrt_alphas_cumprod", to_torch(np.sqrt(train_alphas_cumprod)).to(self.device))
        self.register_buffer("train_sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - train_alphas_cumprod)).to(self.device))
        self.register_buffer("train_sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod)).to(self.device))
        self.register_buffer("train_sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod - 1)).to(self.device))
        self.register_buffer("train_snr", to_torch(train_alphas_cumprod / (1 - train_alphas_cumprod)).to(self.device))

        self.cross_align = nn.ModuleList([
            CrossAlign(ch, ch) for ch in [3, 64, 128, 192]
        ])
        self.feature_dims = [3, 64, 128, 192]

        self.discriminator_w = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(ch, 1, 1)
            )
            for ch in self.feature_dims
        ])

        z_x_channels = [3, 64, 128, 192]
        w_channels = [3, 64, 128, 192]

        if config.fusion == "f":
            self.adain_fusion = MultiScaleSpatialFusion(z_x_channels, w_channels)
        else:
            self.adain_fusion = None

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "loss_fn_vgg" not in name:
                yield param

    def get_discriminator_params(self):
        params = []
        for d in self.discriminator_w:
            params.extend(d.parameters())
        return params

    def get_main_params(self):
        disc_param_ids = set(id(p) for p in self.get_discriminator_params())
        for name, param in self.named_parameters():
            if "loss_fn_vgg" not in name and id(param) not in disc_param_ids:
                yield param

    def get_extra_loss(self):
        return self.context_fn.get_extra_loss()

    def get_effective_beta(self, epoch):
        """Rate Warmup: 前 rate_warmup_epochs 个 epoch 线性增长 beta"""
        if epoch < self.rate_warmup_epochs:
            return self.lagrangian_beta * (epoch / self.rate_warmup_epochs)
        return self.lagrangian_beta

    def update_beta(self, new_beta):
        """供 trainer 在训练中动态更新 beta（自适应 β 用）"""
        self.lagrangian_beta = float(new_beta)

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        self.index = torch.arange(self.num_timesteps, device=device)[indice]
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        ) * torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        self.snr = self.train_snr[indice]

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise_train(self, x_t, t, noise):
        return (
            extract(self.train_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.train_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, x, t, context, clip_denoised):
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t.float().unsqueeze(-1) / self.sample_steps, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            t_input = t.float().unsqueeze(-1) / self.sample_steps
            x_recon = self.denoise_fn(x, t_input, context=context)

        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)

        model_mean = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean

    def ddim(self, x, t, context, contro, clip_denoised, eta=0):
        t_input = t.float().unsqueeze(-1) / self.sample_steps
        fx = self.denoise_fn(x, t_input, context=context, contro=contro)
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t=t, noise=fx)
        elif self.pred_mode == "x":
            x_recon = fx

        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)

        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(x, t=t, x0=x_recon)
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                (extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                 - (eta * extract(self.sigma, t, x.shape)) ** 2).clamp(min=0)
            )
            * noise + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    def p_sample(self, x, t, context, contro, clip_denoised, sample_mode="ddim", eta=0):
        if sample_mode == "ddpm":
            model_mean = self.p_mean_variance(x=x, t=t, context=context, clip_denoised=clip_denoised)
            return model_mean
        elif sample_mode == "ddim":
            return self.ddim(x=x, t=t, context=context, contro=contro, clip_denoised=clip_denoised, eta=eta)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def p_sample_loop(self, shape, context, contro, sample_mode, init=None, eta=0):
        device = self.alphas_cumprod.device
        b = shape[0]
        if config.init == "noise":
            img = torch.randn(shape, device=device) if init is None else init
        else:
            img = context[0]

        for count, i in enumerate(reversed(range(0, self.sample_steps))):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img, time,
                context=contro, contro=contro,
                clip_denoised=self.clip_noise,
                sample_mode=sample_mode,
                eta=eta,
            )
        return img

    def compress(self, images_x, images_y, sample_steps=None, bitrate_scale=None,
                 sample_mode="ddim", bpp_return_mean=True, init=None, eta=0):
        y_w = self.encode_w(images_y)
        w = y_w["output"]
        z_y = self.context_w_p(images_y)["output"]
        context_dict_x = self.context_fn(images_x, bitrate_scale)

        if config.fusion == "add":
            result_x_y = [x + y for x, y in zip(context_dict_x["output"], w)]
        else:
            result_x_y = self.adain_fusion(context_dict_x["output"], w)

        result_x = context_dict_x["output"]

        self.set_sample_schedule(sample_steps, context_dict_x["output"][0].device)

        return (
            self.p_sample_loop(images_x.shape, result_x, result_x_y,
                               sample_mode=sample_mode, init=init, eta=eta),
            result_x[0],
            w[0],
            z_y[0],
            context_dict_x["bpp"] + y_w["bpp"],
            context_dict_x["bpp"]
        )

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, context_dict, w, t, step=None):
        """
        [关键修改] BPP loss 已从这里移除（避免与 forward 中重复加权）
        统一在 forward() 中计算 BPP loss
        """
        num_steps = getattr(self, "unroll_steps", 1)
        detach_unroll = getattr(self, "detach_unroll", False)

        # w 正则项
        w_reg_loss = 0.0
        for w_tensor in w:
            w_reg_loss += torch.mean(w_tensor ** 2)
        w_reg_loss = w_reg_loss / max(1, len(w))

        # 融合特征预计算
        if config.fusion == "add":
            fused_context = [x + y for x, y in zip(context_dict["output"], w)]
        else:
            fused_context = self.adain_fusion(context_dict["output"], w)

        x_current = x_start
        total_diffusion_loss = 0.0
        step_interval = max(1, max(1, getattr(self, "num_timesteps", 1000)) // (num_steps + 1))

        for step_id in range(num_steps):
            if step_id == 0:
                t_i = t
            else:
                t_i = torch.clamp(t - step_id * step_interval, min=0)

            t_input = t_i.float() / self.num_timesteps
            if t_input.ndim == 1:
                t_input = t_input.unsqueeze(-1)

            noise = torch.randn_like(x_current)
            x_noisy = self.q_sample(x_start=x_current, t=t_i, noise=noise)

            fx = self.denoise_fn(x_noisy, t_input, context=fused_context)

            # --- Loss Calculation ---
            if self.pred_mode == "noise":
                weight = (self.train_snr[t_i].clamp(max=self.loss_weight_min) / self.train_snr[t_i])
                if self.loss_type == "l1":
                    err = F.l1_loss(noise, fx, reduction='none').mean(dim=(1, 2, 3))
                    err = (err * torch.sqrt(weight)).mean()
                elif self.loss_type == "l2":
                    err = F.mse_loss(noise, fx, reduction='none').mean(dim=(1, 2, 3))
                    err = (err * weight).mean()
            elif self.pred_mode == "x":
                weight = (self.train_snr[t_i].clamp(max=self.loss_weight_min))
                if self.loss_type == "l1":
                    err = F.l1_loss(x_current, fx, reduction='none').mean(dim=(1, 2, 3))
                    err = (err * torch.sqrt(weight)).mean()
                elif self.loss_type == "l2":
                    err = F.mse_loss(x_current, fx, reduction='none').mean(dim=(1, 2, 3))
                    err = (err * weight).mean()
            else:
                raise NotImplementedError()

            # --- Aux Loss (Perceptual) ---
            aux_err = 0.0
            if self.aux_loss_weight > 0:
                if self.pred_mode == "noise":
                    pred_x0 = self.predict_start_from_noise_train(x_noisy, t_i, fx).clamp(-1.0, 1.0)
                elif self.pred_mode == "x":
                    pred_x0 = fx

                weight_aux = self.train_snr[t_i].clamp(max=self.loss_weight_min)

                if self.aux_loss_type == "l1":
                    aux_err = (torch.sqrt(weight_aux) * F.l1_loss(x_current, pred_x0, reduction='none').mean(dim=(1, 2, 3))).mean()
                elif self.aux_loss_type == "l2":
                    aux_err = (weight_aux * F.mse_loss(x_current, pred_x0, reduction='none').mean(dim=(1, 2, 3))).mean()
                elif self.aux_loss_type == "lpips":
                    aux_err = (weight_aux * self.loss_fn_vgg(x_current, pred_x0).view(x_current.shape[0], -1).mean(dim=1)).mean()

            if self.aux_loss_weight > 0:
                step_loss = err * (1 - self.aux_loss_weight) + aux_err * self.aux_loss_weight
            else:
                step_loss = err

            total_diffusion_loss = total_diffusion_loss + step_loss

            if self.pred_mode == "noise":
                x_pred = self.predict_start_from_noise_train(x_noisy, t_i, fx)
            else:
                x_pred = fx

            x_pred = x_pred.clamp(-1.0, 1.0)

            if detach_unroll:
                x_current = x_pred.detach()
            else:
                x_current = x_pred

        avg_diffusion_loss = total_diffusion_loss / float(num_steps)

        # ===== [关键] 不再加 BPP loss =====
        final_loss = avg_diffusion_loss + 0.01 * w_reg_loss

        return final_loss

    def forward(self, images_x, image_y, step=None, epoch=0):
        if config.load_model:
            epoch = 100
        device = image_y.device

        # ==========================================
        # 0. 阶段 warmup
        # ==========================================
        w_vae = 1.0
        w_diff = min(1.0, max(0.0, (epoch - 5) / 5.0))
        w_aux  = min(1.0, max(0.0, (epoch - 10) / 5.0))

        # Rate Warmup: 基于当前 self.lagrangian_beta（可能已被 trainer 动态更新）
        effective_beta = self.get_effective_beta(epoch)

        # ==========================================
        # 1. VAE 特征提取
        # ==========================================
        z_x = self.context_fn(images_x)
        x_features = z_x.get("output", None)
        bpp = z_x.get("bpp", 0)

        if not isinstance(x_features, (tuple, list)):
            x_features_list = [x_features]
        else:
            x_features_list = x_features

        y_w = self.encode_w(image_y)["output"]
        if not isinstance(y_w, (tuple, list)): y_w = [y_w]

        x_w = self.encode_w(images_x)["output"]
        if not isinstance(x_w, (tuple, list)): x_w = [x_w]

        z_y = self.context_w_p(image_y)["output"]
        if not isinstance(z_y, (tuple, list)): z_y = [z_y]

        # 融合
        if config.fusion == "add":
            fused_features = x_features_list
        else:
            fused_features = self.adain_fusion(x_features_list, y_w)

        # --- VAE 重建 Loss ---
        if config.fusion == "f":
            loss_fused_l1 = F.l1_loss(fused_features[0], images_x)
            loss_fused_lpip = self.loss_fn_vgg(fused_features[0], images_x).mean()
            loss_zx_l1 = F.l1_loss(x_features_list[0], images_x)
            loss_zx_lpip = self.loss_fn_vgg(x_features_list[0], images_x).mean()
            loss_vae = (
                0.6 * (loss_fused_l1 * 0.4 + loss_fused_lpip * 0.3) +
                0.4 * (loss_zx_l1 * 0.4 + loss_zx_lpip * 0.3)
            )
        else:
            loss_vae_recon_l1 = F.l1_loss(fused_features[0], images_x)
            loss_vae_recon_lpip = self.loss_fn_vgg(fused_features[0], images_x).mean()
            loss_vae = loss_vae_recon_l1 * 0.4 + loss_vae_recon_lpip * 0.3

        # ===== [关键] 唯一的 BPP loss =====
        loss_bpp = effective_beta * bpp.mean()
        total_vae_loss = loss_vae + loss_bpp

        # ==========================================
        # 2. 扩散模型（不再包含 BPP）
        # ==========================================
        z_x_input = z_x
        t = torch.randint(0, self.num_timesteps, (images_x.size(0),), device=device)

        raw_loss_x = self.p_losses(images_x, z_x_input, y_w, t)
        final_loss_x = raw_loss_x * w_diff

        # ==========================================
        # 3. 辅助任务 (GAN/Align/Orth)
        # ==========================================
        info_suppression_loss = 0.0
        align_loss = 0.0
        orthogonal_loss = 0.0

        valid_pairs = min(len(x_features_list), len(x_w))
        for i in range(valid_pairs):
            x_feat = x_features_list[i]
            wi = y_w[i]
            zy_i = z_y[i] if i < len(z_y) else z_y[-1]
            xi = x_w[i]

            x_rev = grad_reverse(x_feat, lambda_=0.2)
            w_pred = self.discriminator_w[i](x_rev)
            info_suppression_loss += F.mse_loss(w_pred, xi.detach())

            wi_pred = self.cross_align[i](xi, wi.detach())
            cos_loss = 1 - F.cosine_similarity(wi_pred.flatten(1), wi.detach().flatten(1), dim=1).mean()
            mse_loss = F.mse_loss(wi_pred, wi.detach())
            align_loss += 0.7 * cos_loss + 0.3 * mse_loss

            orthogonal_loss += torch.mean(torch.abs(F.cosine_similarity(wi.flatten(1), zy_i.flatten(1), dim=1)))

        info_suppression_loss /= max(1, valid_pairs)
        align_loss /= max(1, valid_pairs)
        orthogonal_loss /= max(1, valid_pairs)

        total_aux_loss = (
            info_suppression_loss * 0.05 +
            align_loss * 0.05 +
            orthogonal_loss * 0.05
        ) * w_aux

        # 只关闭 AID 的三项辅助训练目标，不改变 VAE、码率、扩散、MSFM、
        # fixed-step/unroll 或任何 warmup。乘以零而不是跳过分支，是为了保持
        # 与完整模型一致的计算图和参数使用关系，确保原有 DDP 训练流程稳定。
        if self.aid_ablation:
            total_aux_loss = total_aux_loss * 0.0

        # ==========================================
        # 4. 汇总
        # ==========================================
        total_loss = total_vae_loss + final_loss_x + total_aux_loss

        # ===== [新增] 返回当前 BPP，供 trainer 自适应 β 更新使用 =====
        current_bpp = bpp.mean().detach()

        return total_loss, self.get_extra_loss(), current_bpp


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
