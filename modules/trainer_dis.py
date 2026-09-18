import torch
from pathlib import Path
from torch.optim import Adam, AdamW
import copy
import math
import numpy as np
from .components import cycle
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from pytorch_msssim import ms_ssim
import time
import os
from PIL import Image
import sys

current_path = Path(__file__).resolve()
parent_dir_a_path = current_path.parent
parent_dir_path = parent_dir_a_path.parent
sys.path.append(str(parent_dir_path))
import config


def save_image(x_recon, x_c, y, z_y, x, path, name):
    """
    训练时 sanity check 用的特征可视化（速度优先）。
    像素图像：直接转 uint8
    特征图：按通道独立归一化
    """
    batch_size = x_recon.size(0)
    dirs = ['img_recon', 'x_c', 'y', 'z_y', 'img']
    for dir_name in dirs:
        os.makedirs(os.path.join(path, dir_name), exist_ok=True)

    def to_pixel_image(tensor_img):
        arr = (tensor_img * 255).clamp(0, 255).byte().cpu().numpy()
        return np.transpose(arr, (1, 2, 0))

    def to_feature_image(feat_tensor):
        feat = feat_tensor.detach().cpu().float()
        C, H, W = feat.shape
        
        # 通道数处理
        if C >= 3:
            vis = feat[:3]
        else:
            vis = feat[0:1].repeat(3, 1, 1)
        
        # 按通道独立归一化（快速，训练时够用）
        vis_normalized = torch.zeros_like(vis)
        for c in range(3):
            ch = vis[c]
            ch_min = ch.min()
            ch_max = ch.max()
            if (ch_max - ch_min) > 1e-6:
                vis_normalized[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                vis_normalized[c] = 0.5
        
        arr = (vis_normalized * 255).clamp(0, 255).byte().numpy()
        return np.transpose(arr, (1, 2, 0))

    for i in range(batch_size):
        img_recon = to_pixel_image(x_recon[i])
        img_c = to_pixel_image(x_c[i])
        img = to_pixel_image(x[i])
        y_img = to_feature_image(y[i])
        z_y_img = to_feature_image(z_y[i])

        filename = f"{name}_{i}.png"

        Image.fromarray(img_recon).save(os.path.join(path, 'img_recon', filename))
        Image.fromarray(img_c).save(os.path.join(path, 'x_c', filename))
        Image.fromarray(y_img).save(os.path.join(path, 'y', filename))
        Image.fromarray(z_y_img).save(os.path.join(path, 'z_y', filename))
        Image.fromarray(img).save(os.path.join(path, 'img', filename))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


class Trainer(object):
    def __init__(
        self,
        accelerator,
        rank,
        sample_steps,
        diffusion_model,
        train_loader,
        val_loader,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm",
        lagrangian=1,
        disc_lr=2.5e-5,
    ):
        super().__init__()
        self.model = diffusion_model
        self.sample_mode = sample_mode
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every
        self.accelerator = accelerator
        self.train_num_steps = train_num_steps

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lagrangian = lagrangian

        # ===== 主网络和判别器分开 optimizer =====
        model_unwrapped = accelerator.unwrap_model(diffusion_model) if hasattr(accelerator, 'unwrap_model') else diffusion_model

        main_params = list(model_unwrapped.get_main_params())
        if optimizer == "adam":
            self.opt = Adam(main_params, lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(main_params, lr=train_lr)

        disc_params = list(model_unwrapped.get_discriminator_params())
        self.opt_disc = Adam(disc_params, lr=disc_lr, betas=(0.5, 0.999))

        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)
        self.scheduler_disc = LambdaLR(self.opt_disc, lr_lambda=scheduler_function)

        # ===== Model EMA（用于推理）=====
        self.ema_decay = ema_decay
        self.ema = EMA(ema_decay)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every

        # ===== 自适应 β 控制器 =====
        # [关键] 改用验证时的真实 transmitted_bpp 来更新 β
        # 训练时的 bpp（用 noise 量化估计）会被模型"作弊"
        # 真实 transmitted_bpp（dequantize 量化）才是模型实际的码率状态
        self.target_bpp = getattr(config, 'target_bpp', None)
        self.beta_lr = getattr(config, 'beta_lr', 0.1)
        self.beta_min = getattr(config, 'beta_min', 1e-3)
        self.beta_max = getattr(config, 'beta_max', 1.0)   # 上限改为 1.0，避免 z_x 完全废掉
        self.current_beta = config.beta
        self.rate_warmup_epochs = getattr(config, 'rate_warmup_epochs', 50)

        self.step = 0
        self.global_step = 0
        self.device = accelerator.device
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

        (self.model, self.opt, self.opt_disc,
         self.train_loader, self.val_loader,
         self.scheduler, self.scheduler_disc) = self.accelerator.prepare(
            self.model, self.opt, self.opt_disc,
            self.train_loader, self.val_loader,
            self.scheduler, self.scheduler_disc
        )

        self._ema_initialized = False

    def _init_ema_model(self):
        if not self._ema_initialized:
            model_unwrapped = self.accelerator.unwrap_model(self.model)
            self.ema_model = copy.deepcopy(model_unwrapped)
            self.ema_model.eval()
            self._ema_initialized = True

    def _update_ema(self):
        if not self._ema_initialized:
            self._init_ema_model()
        model_unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema.update_model_average(self.ema_model, model_unwrapped)

    def _build_hparams(self):
        return {
            "embed_dim": config.embed_dim,
            "dim_mults": config.dim_mults,
            "context_dim_mults": config.context_dim_mults,
            "hyper_dim_mults": config.hyper_dim_mults,
            "context_channels": config.context_channels,
            "img_channel": config.data_config["img_channel"],
            "compressor_": config.compressor_,
            "fusion": config.fusion,
            "iteration_step": config.iteration_step,
            "sample_steps": config.sample_steps,
            "pred_mode": config.pred_mode,
            "var_schedule": config.var_schedule,
            "clip_noise": config.clip_noise,
            "init": config.init,
            "beta": config.beta,
            "alpha": config.alpha,
            "aux_loss_type": config.aux_loss_type,
            "loss_type": config.loss_type,
            "vbr": config.vbr,
            "unroll_steps": config.unroll_steps,
            "detach_unroll": config.detach_unroll,
            # 将消融状态写入 checkpoint，避免实验结果与配置身份混淆。
            "aid_ablation": getattr(config, "aid_ablation", False),
            "dataset_name": config.data_config["dataset_name"],
            "data_path": config.data_config["data_path"],
            "use_full_attn_mid": getattr(config, 'use_full_attn_mid', True),
            "target_bpp": self.target_bpp,
            "final_beta": self.current_beta,
        }

    def save(self, suffix="best"):
        model = self.accelerator.unwrap_model(self.model)
        data = {
            "step": self.step,
            "model": model.state_dict(),
            "hparams": self._build_hparams(),
            "current_beta": self.current_beta,
        }
        if self._ema_initialized:
            data["ema"] = self.ema_model.state_dict()

        self.accelerator.save(data, str(self.results_folder / f"{self.model_name}_{suffix}.pt"))

    def save_checkpoint(self):
        model = self.accelerator.unwrap_model(self.model)
        data = {
            "step": self.step,
            "model": model.state_dict(),
            "opt": self.opt.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "hparams": self._build_hparams(),
            "current_beta": self.current_beta,
        }
        if self._ema_initialized:
            data["ema"] = self.ema_model.state_dict()

        idx = (self.step // self.save_and_sample_every) % 3
        self.accelerator.save(data, str(self.results_folder / f"{self.model_name}_ckpt_{idx}.pt"))

    def _adaptive_beta_update(self, real_transmitted_bpp, epoch):
        """
        使用验证时的真实 transmitted_bpp（dequantize 量化）来更新 β。

        关键：训练时的 bpp 用 noise 量化估计，可能被模型"作弊"得到虚高估计；
        验证时的 transmitted_bpp 才是模型真实的传输码率，这才是 β 的目标信号。

        每次验证才调用一次（频率低、信号稳定），用对数空间更新。
        """
        if self.target_bpp is None:
            return  # 禁用自适应

        if epoch < self.rate_warmup_epochs:
            return  # warmup 期间用固定 β

        if real_transmitted_bpp <= 0 or real_transmitted_bpp > 100:
            return  # 防御异常

        # 对数空间更新
        ratio = real_transmitted_bpp / self.target_bpp
        # 放宽 ratio 范围（因为更新频率低，每次可以多调一点）
        ratio = max(0.2, min(5.0, ratio))

        log_beta_delta = self.beta_lr * math.log(ratio)
        new_beta = self.current_beta * math.exp(log_beta_delta)
        new_beta = max(self.beta_min, min(self.beta_max, new_beta))

        old_beta = self.current_beta
        self.current_beta = new_beta
        self.accelerator.unwrap_model(self.model).update_beta(new_beta)

        return old_beta, new_beta

    def train(self):
        mse = torch.nn.MSELoss(reduction='mean')
        mse = mse.to(self.device)
        st = time.time()
        min_loss = float('inf')

        for epoch in range(self.train_num_steps):
            self.model.train()
            self.step = epoch

            if (epoch % self.scheduler_checkpoint_step == 0) and (epoch != 0):
                self.scheduler.step()
                self.scheduler_disc.step()

            for i, data in enumerate(iter(self.train_loader)):
                self.opt.zero_grad()
                self.opt_disc.zero_grad()

                img, cor_img, _, _ = data
                img = img.float().to(self.device)
                cor_img = cor_img.float().to(self.device)

                # forward 仍然返回 3 个值（保持模型接口不变），但不再用 current_bpp 更新 β
                loss_x, aloss_x, current_bpp = self.model(
                    img * 2.0 - 1.0, cor_img * 2.0 - 1.0, epoch=epoch
                )

                self.accelerator.backward(loss_x + aloss_x)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.opt.step()
                self.opt_disc.step()

                self.global_step += 1

                # Model EMA 更新
                if self.global_step >= self.step_start_ema and self.global_step % self.update_ema_every == 0:
                    self._update_ema()

            # ==========================================
            # 验证 & 保存 & β 更新
            # ==========================================
            if (epoch % self.save_and_sample_every == 0 or epoch == 11):

                val_loss = []
                val_mse = []
                val_msssim = []
                val_bpp = []
                val_transmitted_bpp = []
                val_distortion = []

                self.model.eval()
                with torch.no_grad():
                    for i, data in enumerate(iter(self.val_loader)):
                        img, cor_img, _, _ = data
                        img = img.float().to(self.device)
                        cor_img = cor_img.float().to(self.device)

                        compressed_x, x, y, z_y, bpp_val, transmitted_bpp = self.accelerator.unwrap_model(self.model).compress(
                            img * 2.0 - 1.0,
                            cor_img * 2.0 - 1.0,
                            sample_steps=config.sample_steps
                        )
                        compressed_x = compressed_x.clamp(-1, 1) / 2.0 + 0.5

                        mse_dist = mse(img, compressed_x)
                        msssim = ms_ssim(img.clone(), compressed_x.clone(), data_range=1.0, size_average=True, win_size=7)
                        msssim_db = msssim

                        distortion = (1 - ms_ssim(img, compressed_x, data_range=1.0, size_average=True, win_size=7))

                        loss = self.lagrangian * distortion * (255 ** 2) + transmitted_bpp

                        if self.accelerator.is_main_process and i == 0:
                            out_dir = os.path.join('./output', self.model_name)
                            os.makedirs(out_dir, exist_ok=True)
                            save_image(compressed_x, x, y, z_y, img, out_dir, str(i))

                        val_transmitted_bpp.append(torch.mean(transmitted_bpp).to(self.accelerator.device))
                        val_bpp.append(torch.mean(bpp_val).to(self.accelerator.device))
                        val_mse.append(mse_dist.to(self.accelerator.device))
                        val_loss.append(torch.mean(loss).to(self.accelerator.device))
                        val_msssim.append(torch.mean(msssim_db).to(self.accelerator.device))
                        val_distortion.append(torch.mean(distortion).to(self.accelerator.device))

                gathered_val_loss = self.accelerator.gather_for_metrics(torch.stack(val_loss))
                gathered_val_mse = self.accelerator.gather_for_metrics(torch.stack(val_mse))
                gathered_val_transmitted_bpp = self.accelerator.gather_for_metrics(torch.stack(val_transmitted_bpp))
                gathered_val_bpp = self.accelerator.gather_for_metrics(torch.stack(val_bpp))
                gathered_val_msssim = self.accelerator.gather_for_metrics(torch.stack(val_msssim))
                gathered_val_distortion = self.accelerator.gather_for_metrics(torch.stack(val_distortion))

                val_loss_to_track = gathered_val_loss.mean().item()
                avg_bpp = gathered_val_bpp.mean().item()
                avg_t_bpp = gathered_val_transmitted_bpp.mean().item()  # ← 真实 transmitted_bpp
                avg_distortion = gathered_val_distortion.mean().item()
                avg_mse = gathered_val_mse.mean().item()
                avg_msssim = gathered_val_msssim.mean().item()
                avg_psnr = 10 * np.log10(1 / avg_mse)

                # ===== [关键] 用真实 transmitted_bpp 更新 β =====
                old_beta = self.current_beta
                update_result = self._adaptive_beta_update(avg_t_bpp, epoch)
                beta_changed = update_result is not None and abs(update_result[1] - update_result[0]) > 1e-9

                if self.target_bpp is not None:
                    # 绝对偏差
                    bpp_abs_deviation = abs(avg_t_bpp - self.target_bpp)
                    # 相对偏差
                    bpp_rel_deviation = bpp_abs_deviation / self.target_bpp
                    
                    # 缓冲带：相对 25% 或 绝对 0.02 取较大者
                    # 低 target 用绝对值（避免过于苛刻）
                    # 高 target 用相对值（保持比例性）
                    abs_tolerance = 0.025      # 绝对容忍：±0.02
                    rel_tolerance = 0.25      # 相对容忍：±25%
                    
                    # 偏差超过容忍范围才惩罚
                    effective_dev = max(0, 
                        bpp_rel_deviation - rel_tolerance,
                        (bpp_abs_deviation - abs_tolerance) / max(self.target_bpp, 0.05)
                    )
                    bpp_penalty = effective_dev
                    
                    rd_score = avg_distortion + bpp_penalty * 0.5
                else:
                    rd_score = val_loss_to_track
                    bpp_rel_deviation = 0.0

                tracking = ['Epoch {}:'.format(epoch + 1),
                            'Loss= {:.4f},'.format(val_loss_to_track),
                            'BPP= {:.4f},'.format(avg_bpp),
                            'Distortion= {:.4f},'.format(avg_distortion),
                            'Transmitted BPP = {:.4f},'.format(avg_t_bpp),
                            'PSNR= {:.4f},'.format(avg_psnr),
                            'MS-SSIM= {:.4f}'.format(avg_msssim)]

                end_time = time.time()
                execution_time = end_time - st
                formatted_time = format_duration(execution_time)

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    lr_main = self.opt.param_groups[0]['lr']
                    lr_disc = self.opt_disc.param_groups[0]['lr']

                    # 显示 β 更新信息
                    if beta_changed:
                        beta_info = f"β:{old_beta:.6f}->{self.current_beta:.6f}"
                    else:
                        beta_info = f"β={self.current_beta:.6f}"

                    if self.target_bpp is not None:
                        beta_info += f" target={self.target_bpp}"

                    print(f"{formatted_time} lr_main={lr_main:.6f} lr_disc={lr_disc:.6f} {beta_info}  " + " ".join(tracking))

                    if rd_score < min_loss:
                        min_loss = rd_score
                        self.save(suffix="best")
                        if self.target_bpp is not None:
                            print(f"  >> New best model saved! "
                                f"rd_score={rd_score:.4f} "
                                f"(distortion={avg_distortion:.4f}, bpp_dev={bpp_abs_deviation*100:.1f}%), "
                                f"β={self.current_beta:.6f}")
                        else:
                            print(f"  >> New best model saved! val_loss={val_loss_to_track:.4f}, β={self.current_beta:.6f}")

                    self.save_checkpoint()

                self.accelerator.wait_for_everyone()

        print("training completed")
