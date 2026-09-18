"""
test.py - 自包含的测试脚本
所有超参数从 checkpoint 读取,不再依赖 config.py
新增指标: FID, SSIM (在原有 BPP, PSNR, MS-SSIM, DISTS, LPIPS 基础上)

FID 计算与旧版手写脚本逐位对齐:
  - inception_v3(pretrained=True, transform_input=False), fc=Identity, eval()
  - 输入路径: tensor [0,1] -> PIL Image (uint8 量化, 等价于 PNG save/load)
    -> transforms.Resize((299, 299)) -> ToTensor() -> Normalize(ImageNet)
  - mu/sigma 用 numpy, 协方差矩阵平方根用 scipy.linalg.sqrtm, 复数取实部
"""
import time
import argparse
import os
import torch
import numpy as np
import pandas as pd
import lpips
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim, ssim
from DISTS_pytorch import DISTS
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

# 模型导入
from modules.model import DICM
from modules.unet import Unet
from modules.compressor import Compressor, ResnetCompressor, f_w_compressor
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape


# ============================================================
# 参数解析(只需要 ckpt 路径和少量运行时参数)
# ============================================================
parser = argparse.ArgumentParser(description="DICM Test - config-free, loads everything from checkpoint")
parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
parser.add_argument("--device", type=int, default=0, help="GPU device index")
parser.add_argument("--data_path", type=str, default=None,
                    help="Override dataset path (default: use path from checkpoint)")
parser.add_argument("--out_dir", type=str, default='./result', help="Output directory")
parser.add_argument("--sample_steps", type=int, default=None,
                    help="Override sample steps (default: use value from checkpoint)")
parser.add_argument("--batch_size", type=int, default=1, help="Test batch size")
parser.add_argument("--use_ema", type=int, default=0, help="Use EMA weights if available")
parser.add_argument("--compute_fid", type=int, default=1, help="Compute FID (slower, needs full dataset)")
args = parser.parse_args()


# ============================================================
# FID 计算器 (与旧版手写脚本逐位对齐)
# ============================================================
class FIDCalculator:
    """
    与旧版手写 FID 实现保持一致的实现。

    关键点:
      1. inception_v3(pretrained=True, transform_input=False), fc -> Identity, eval
      2. 输入路径走 PIL: tensor[0,1] -> to_pil_image(uint8 量化)
         -> Resize((299, 299)) -> ToTensor -> Normalize(ImageNet)
         这一步刻意走 PIL 是为了与旧脚本 "从磁盘读 PNG 再 transform" 的路径
         保持像素级一致(PNG 是无损的, 但保存/读取必经过 uint8 量化)。
      3. mu = features.mean(0), sigma = np.cov(features, rowvar=False)
      4. covmean = scipy.linalg.sqrtm(sigma1 @ sigma2), 复数则取实部
      5. FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * covmean)

    接口与 torchmetrics.FrechetInceptionDistance 保持一致:
      fid.update(images, real=True/False)   # images in [0, 1], shape [B, 3, H, W]
      fid.compute()                          # -> float
    """

    # 与旧脚本完全相同的 transform
    _PIL_TRANSFORM = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, device=0):
        self.device = device

        # 兼容新旧版 torchvision 的权重加载方式
        try:
            from torchvision.models import Inception_V3_Weights
            model = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
        except ImportError:
            model = inception_v3(pretrained=True, transform_input=False)

        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.cuda(device)

        self.real_feats = []
        self.fake_feats = []

    def _preprocess_via_pil(self, x):
        """
        x: [B, 3, H, W] float tensor in [0, 1]
        return: [B, 3, 299, 299] tensor on GPU, ImageNet-normalized

        刻意走 PIL 路径以匹配旧脚本 "保存 PNG -> 读取 PNG -> transform" 的行为。
        to_pil_image 会自动做 *255 + uint8 量化, 等价于 PNG 无损保存读取。
        """
        x = x.detach().clamp(0, 1).cpu()
        out = []
        for i in range(x.shape[0]):
            # tensor[3,H,W] in [0,1] -> uint8 PIL Image (等价于 PNG round-trip)
            pil_img = TF.to_pil_image(x[i])
            # 与旧脚本完全相同的 transform
            tensor_out = self._PIL_TRANSFORM(pil_img)
            out.append(tensor_out)
        return torch.stack(out, dim=0).cuda(self.device, non_blocking=True)

    @torch.no_grad()
    def update(self, images, real=True):
        """images: [B, 3, H, W] in [0, 1]"""
        x = self._preprocess_via_pil(images)
        feats = self.model(x).detach().cpu()
        (self.real_feats if real else self.fake_feats).append(feats)

    def compute(self):
        real = torch.cat(self.real_feats, dim=0).numpy()
        fake = torch.cat(self.fake_feats, dim=0).numpy()

        mu1, sigma1 = real.mean(0), np.cov(real, rowvar=False)
        mu2, sigma2 = fake.mean(0), np.cov(fake, rowvar=False)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def reset(self):
        self.real_feats.clear()
        self.fake_feats.clear()


# ============================================================
# 工具函数
# ============================================================
def save_image(x_recon, x, zx, w, z_y, path, name):
    """
    保存原图、重建图、中间特征图(论文质量)。

    - 像素域图像(x, x_recon, zx):直接 [0,1] -> uint8
    - 特征图(w, z_y):PCA 降维到 3 通道再归一化,保留最大方差
    """
    def to_pixel_numpy(tensor_img):
        """像素域:[0,1] 范围直接转 uint8"""
        if tensor_img.dim() == 4:
            tensor_img = tensor_img.squeeze(0)
        arr = np.clip((tensor_img * 255).cpu().numpy(), 0, 255)
        arr = np.transpose(arr, (1, 2, 0)).astype('uint8')
        return arr

    def feature_to_numpy(feat_tensor):
        """
        特征图可视化(论文质量):
        1. PCA 降维到 3 通道(保留最大方差)
        2. 全图统一归一化(保持空间一致性,避免色彩跳变)
        3. 适度 clip 极端值避免单点拉伸整图
        """
        if feat_tensor.dim() == 4:
            feat_tensor = feat_tensor.squeeze(0)

        feat = feat_tensor.detach().cpu().float()  # [C, H, W]
        C, H, W = feat.shape

        if C == 3:
            # 已经是 3 通道,直接归一化
            vis = feat
        else:
            # PCA 降维到 3 通道
            # reshape 成 [HW, C],每行是一个像素的 C 维特征
            feat_flat = feat.permute(1, 2, 0).reshape(-1, C)  # [HW, C]
            # 中心化
            feat_centered = feat_flat - feat_flat.mean(dim=0, keepdim=True)
            # SVD 求主成分
            try:
                U, S, V = torch.linalg.svd(feat_centered, full_matrices=False)
                # 取前 3 个主成分方向
                principal = V[:3].T  # [C, 3]
                # 投影
                projected = feat_centered @ principal  # [HW, 3]
                vis = projected.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
            except Exception:
                # SVD 失败时退化成取前 3 通道
                vis = feat[:3] if C >= 3 else feat[0:1].repeat(3, 1, 1)

        # 归一化:先 clip 到 [1%, 99%] 分位数避免极端值,再线性映射到 [0, 1]
        vis_np = vis.numpy()
        lo = np.percentile(vis_np, 1)
        hi = np.percentile(vis_np, 99)
        if hi - lo > 1e-6:
            vis_np = np.clip((vis_np - lo) / (hi - lo), 0, 1)
        else:
            vis_np = np.zeros_like(vis_np) + 0.5

        arr = (vis_np * 255).astype('uint8')
        arr = np.transpose(arr, (1, 2, 0))  # HWC
        return arr

    # 像素域图像(x_recon, x, zx 都是已经在 [0,1] 范围的图像)
    # 注:x_recon 经过了 clamp(-1,1)/2 + 0.5,所以在 [0,1]
    #     x 是原图在 [0,1]
    #     zx 是 ResnetCompressor 的输出,可能也在 [0,1] 附近
    pixel_subdirs = {
        'img': x,
        'img_recon': x_recon,
        'zx': zx,
    }

    # 特征图(w, z_y 是网络中间层特征,范围未知)
    feature_subdirs = {
        'w': w,
        'z_y': z_y,
    }

    # 保存像素域图像
    for subdir, tensor in pixel_subdirs.items():
        dir_path = os.path.join(path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        Image.fromarray(to_pixel_numpy(tensor), 'RGB').save(
            os.path.join(dir_path, name + '.png')
        )

    # 保存特征图(PCA 可视化)
    for subdir, tensor in feature_subdirs.items():
        dir_path = os.path.join(path, subdir)
        os.makedirs(dir_path, exist_ok=True)
        Image.fromarray(feature_to_numpy(tensor), 'RGB').save(
            os.path.join(dir_path, name + '.png')
        )


def build_model_from_hparams(hp, device):
    """从超参数字典构建模型(不依赖 config.py)"""

    # 1. UNet
    denoise = Unet(
        dim=hp['embed_dim'],
        channels=hp['img_channel'],
        context_channels=hp['context_channels'],
        dim_mults=tuple(hp['dim_mults']),
        context_dim_mults=tuple(hp['context_dim_mults']),
        use_full_attn_mid=hp.get('use_full_attn_mid', True),
    )

    # 2. X Compressor(带量化)
    context = ResnetCompressor(
        dim=hp['embed_dim'],
        dim_mults=tuple(hp['context_dim_mults']),
        hyper_dims_mults=tuple(hp['hyper_dim_mults']),
        channels=hp['img_channel'],
        out_channels=hp['context_channels'],
    )

    # 3. Y Compressor(bypass,不量化)
    if hp['compressor_'] == 'big':
        context_w = ResnetCompressor(
            dim=hp['embed_dim'],
            dim_mults=tuple(hp['context_dim_mults']),
            hyper_dims_mults=tuple(hp['hyper_dim_mults']),
            channels=hp['img_channel'],
            out_channels=3,
            mode="bypass"
        )
        context_w_p = ResnetCompressor(
            dim=hp['embed_dim'],
            dim_mults=tuple(hp['context_dim_mults']),
            hyper_dims_mults=tuple(hp['hyper_dim_mults']),
            channels=hp['img_channel'],
            out_channels=3,
            mode="bypass"
        )
    elif hp['compressor_'] == 'f':
        context_w = f_w_compressor()
        context_w_p = f_w_compressor()
    else:
        raise ValueError(f"Unknown compressor type: {hp['compressor_']}")

    # 4. DICM
    model = DICM(
        device=device,
        denoise_fn=denoise,
        context_fn=context,
        context_w=context_w,
        context_w_p=context_w_p,
        clip_noise=hp.get('clip_noise', 'half'),
        num_timesteps=hp['iteration_step'],
        loss_type=hp['loss_type'],
        vbr=hp.get('vbr', False),
        lagrangian=hp['beta'],
        pred_mode=hp['pred_mode'],
        aux_loss_weight=hp.get('alpha', 0),
        aux_loss_type=hp.get('aux_loss_type', 'l1'),
        var_schedule=hp['var_schedule'],
        unroll_steps=hp.get('unroll_steps', 1),
        detach_unroll=hp.get('detach_unroll', False),
        # 评测只需复原 checkpoint 的配置身份；旧 checkpoint 没有该字段时按完整模型处理。
        aid_ablation=hp.get('aid_ablation', False),
    )

    return model


def load_checkpoint(ckpt_path, device):
    """加载 checkpoint,返回 (hparams, state_dict, ema_dict_or_None)"""
    print(f"Loading checkpoint: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=f'cuda:{device}')

    # 检查是否包含超参数
    if 'hparams' not in data:
        raise RuntimeError(
            "Checkpoint does not contain 'hparams'. "
            "This checkpoint was saved before the optimization. "
            "Please re-save it or use the old config-based test.py."
        )

    hp = data['hparams']
    state_dict = data['model']
    ema_dict = data.get('ema', None)
    step = data.get('step', 'unknown')

    print(f"  Step: {step}")
    print(f"  Dataset: {hp.get('dataset_name', '?')}")
    print(f"  Beta: {hp['beta']}, Pred mode: {hp['pred_mode']}")
    print(f"  Sample steps: {hp['sample_steps']}")
    print(f"  Compressor: {hp['compressor_']}, Fusion: {hp.get('fusion', '?')}")
    if ema_dict is not None:
        print(f"  EMA weights: available")

    return hp, state_dict, ema_dict


def build_dataloader(hp, args):
    """构建测试 DataLoader"""
    data_path = args.data_path or hp.get('data_path')
    if data_path is None:
        raise ValueError("No data_path found in checkpoint or command line args")

    resize = (128, 256)
    dataset_name = hp['dataset_name']

    if dataset_name == 'KITTI':
        test_dataset = PairKitti(path=data_path, set_type='test', resize=resize)
    elif dataset_name == 'Cityscape':
        test_dataset = PairCityscape(path=data_path, set_type='test', resize=resize)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Dataset: {dataset_name}, size: {len(test_dataset)}")

    return DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )


# ============================================================
# 主流程
# ============================================================
def main():
    device = args.device
    torch.cuda.set_device(device)

    # 1. 加载 checkpoint 和超参数
    hp, state_dict, ema_dict = load_checkpoint(args.ckpt, device)

    # 2. 构建模型
    print("Building model from checkpoint hparams...")
    model = build_model_from_hparams(hp, device)

    # 选择加载 EMA 还是普通权重
    if args.use_ema and ema_dict is not None:
        print("Loading EMA weights...")
        missing, unexpected = model.load_state_dict(ema_dict, strict=False)
    else:
        if args.use_ema and ema_dict is None:
            print("WARNING: --use_ema specified but no EMA weights in checkpoint, using normal weights")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.cuda()
    model.eval()

    # 3. 构建 DataLoader
    print("Loading dataset...")
    test_loader = build_dataloader(hp, args)

    # 4. 构建评估指标
    sample_steps = args.sample_steps or hp['sample_steps']
    print(f"Using sample_steps={sample_steps}")

    mse_fn = torch.nn.MSELoss(reduction='mean').cuda()
    dists_fn = DISTS().cuda()
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()

    # FID 计算器(与旧版手写脚本完全一致, PIL 路径)
    fid_metric = None
    if args.compute_fid:
        print("FID computation enabled (legacy PIL pipeline, slower but matches old results)")
        fid_metric = FIDCalculator(device=device)

    # 5. 输出目录
    model_name = (
        f"{hp['loss_type']}-{hp['dataset_name']}"
        f"-t{hp['iteration_step']}-v{sample_steps}-b{hp['beta']}"
        f"-{hp.get('compressor_', '?')}-{hp['pred_mode']}"
        f"-{int(time.time())}"  # 添加时间戳
    )
    results_path = os.path.join(args.out_dir, 'test', model_name)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results will be saved to: {results_path}")

    # 6. 测试循环
    names = ["Image Number", "BPP", "tran-BPP", "PSNR", "SSIM", "MS-SSIM", "DISTS", "LPIPS"]
    cols = {name: [] for name in names}

    print(f"\n{'='*60}")
    print(f"Starting evaluation...")
    print(f"{'='*60}")

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, cor_img, _, _ = data
            img = img.float().cuda()
            cor_img = cor_img.float().cuda()

            # 压缩 + 重建
            compressed_x, zx, w, z_y, bpp, transmitted_bpp = model.compress(
                img * 2.0 - 1.0,
                cor_img * 2.0 - 1.0,
                sample_steps=sample_steps
            )

            x_recon = compressed_x.clamp(-1, 1) / 2.0 + 0.5

            # --- 计算指标 ---
            # MSE -> PSNR
            mse_dist = mse_fn(img, x_recon)
            psnr_val = 10 * np.log10(1 / mse_dist.item())

            # SSIM(新增)
            ssim_val = ssim(img, x_recon, data_range=1.0, size_average=True).item()

            # MS-SSIM (dB)
            msssim_val = ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True, win_size=7).item()
            msssim_db = -10 * np.log10(1 - msssim_val + 1e-10)

            # DISTS
            dists_val = dists_fn(img, x_recon, require_grad=False, batch_average=True).item()

            # LPIPS
            lpips_val = lpips_fn(img, x_recon).item()

            # FID(逐批累积,最后算)
            if fid_metric is not None:
                fid_metric.update(img, real=True)
                fid_metric.update(x_recon.clamp(0, 1), real=False)

            # --- 记录 ---
            vals = [
                str(i),
                f'{bpp.item():.8f}',
                f'{transmitted_bpp.item():.8f}',
                f'{psnr_val:.4f}',
                f'{ssim_val:.6f}',
                f'{msssim_db:.4f}',
                f'{dists_val:.6f}',
                f'{lpips_val:.6f}',
            ]
            for name, val in zip(names, vals):
                cols[name].append(val)

            # 保存图片(每张都存)
            zx_img = zx.clamp(-1, 1) / 2.0 + 0.5
            w_img = w.clamp(-1, 1) / 2.0 + 0.5
            z_y_img = z_y.clamp(-1, 1) / 2.0 + 0.5

            save_image(
                x_recon[0], img[0], zx_img[0], w_img[0], z_y_img[0],
                os.path.join(results_path, 'DICM_images'), str(i)
            )

            # 打印进度
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(test_loader)}] "
                      f"BPP={bpp.item():.4f} tBPP={transmitted_bpp.item():.4f} "
                      f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} "
                      f"DISTS={dists_val:.4f} LPIPS={lpips_val:.4f}")

    # 7. 汇总结果
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")

    df = pd.DataFrame(cols)

    # 计算均值
    numeric_cols = ["BPP", "tran-BPP", "PSNR", "SSIM", "MS-SSIM", "DISTS", "LPIPS"]
    means = {}
    for col in numeric_cols:
        values = [float(v) for v in df[col]]
        mean_val = np.mean(values)
        means[col] = mean_val
        print(f"  {col:>12s}: {mean_val:.6f}")

    # FID(整体指标)
    if fid_metric is not None:
        fid_val = fid_metric.compute()
        print(f"  {'FID':>12s}: {fid_val:.4f}")
        means['FID'] = fid_val

    # 保存 CSV
    csv_path = os.path.join(results_path, model_name + '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nPer-image results saved to: {csv_path}")

    # 保存均值摘要
    summary_path = os.path.join(results_path, model_name + '_summary.csv')
    pd.DataFrame([means]).to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
