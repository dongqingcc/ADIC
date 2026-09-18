import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.model import DICM
from modules.unet import Unet
import torch
from modules.trainer_dis import Trainer
from modules.compressor import Compressor, w_compressor, big_w_compressor, f_w_compressor, ResnetCompressor
from modules.ramit import RAMiT
import torch.nn as nn
import config
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import random
import numpy as np

# 1. 创建 DDP 配置
if config.load_model:
    find = False
else:
    find = False
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find)
accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=torch.device, default=accelerator.device, help="cuda device number")
args = parser.parse_args()

model_name = (
    f"{config.loss_type}-{config.data_config['dataset_name']}"
    f"-t{config.iteration_step}-v{config.sample_steps}-t{config.target_bpp}-b{config.beta}-vbr{config.vbr}-{config.compressor_}_compressor-u_s{config.unroll_steps}_{config.init}_{config.fusion}"
    f"-{config.pred_mode}-{config.var_schedule}-aux{config.alpha}{config.aux_loss_type if config.alpha > 0 else ''}"
    f"-seed{config.seed}"
    f"{'-aid_ablation' if getattr(config, 'aid_ablation', False) else ''}"
    f"{config.additional_note}"
)


def load_and_freeze_denoise_fn(big_model, ckpt_path="CDC.pt", freeze_first_n=4, device=accelerator.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    original_state_dict = checkpoint["model"]

    denoise_state_dict = {}
    current_model_dict = big_model.denoise_fn.state_dict()
    skipped_keys = []

    for k, v in original_state_dict.items():
        if k.startswith("denoise_fn."):
            if "time_mlp" in k:
                continue
            new_key = k.replace("denoise_fn.", "")
            if new_key in current_model_dict:
                target_shape = current_model_dict[new_key].shape
                if v.shape != target_shape:
                    skipped_keys.append(f"{new_key} (ckpt: {v.shape} -> model: {target_shape})")
                    continue
            denoise_state_dict[new_key] = v

    missing, unexpected = big_model.denoise_fn.load_state_dict(denoise_state_dict, strict=False)

    def freeze_unet_maintain_time(unet, n=4):
        for i in range(min(n, len(unet.downs))):
            is_layer_skipped = any(f"downs.{i}." in sk for sk in skipped_keys)
            for m in unet.downs[i]:
                for p in m.parameters():
                    if is_layer_skipped:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        if n >= len(unet.downs):
            for m in [unet.mid_block1, unet.mid_attn, unet.mid_block2]:
                for p in m.parameters():
                    p.requires_grad = False

        for p in unet.final_conv.parameters():
            p.requires_grad = True

        if hasattr(unet, "time_mlp") and unet.time_mlp is not None:
            for p in unet.time_mlp.parameters():
                p.requires_grad = True

    freeze_unet_maintain_time(big_model.denoise_fn, n=freeze_first_n)

    for module in big_model.denoise_fn.time_mlp.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def load_context_fn_partial_verbose(big_model, ckpt_path="CDC.pt", device=accelerator.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    original_state_dict = checkpoint["model"]
    current_model_dict = big_model.context_fn.state_dict()
    
    load_dict = {}
    report_loaded = []
    report_shape_mismatch = []
    
    for k, v in original_state_dict.items():
        if k.startswith("context_fn."):
            local_key = k.replace("context_fn.", "")
            if local_key in current_model_dict:
                target_shape = current_model_dict[local_key].shape
                if v.shape == target_shape:
                    load_dict[local_key] = v
                    report_loaded.append(local_key)
                else:
                    report_shape_mismatch.append(f"{local_key} (Ckpt: {v.shape} -> Model: {target_shape})")

    missing_keys, unexpected_keys = big_model.context_fn.load_state_dict(load_dict, strict=False)
    return missing_keys


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def load(model, model_name_str, suffix="best", load_step=True):
    """
    从训练保存的 checkpoint 加载（热启动）
    路径: {result_root}/{model_name}/{model_name}_{suffix}.pt

    Args:
        model: 模型
        model_name_str: 模型名（与训练保存路径一致）
        suffix: 'best' / 'best_psnr' / 'ckpt_0' 等
        load_step: 是否加载 step 信息
    """
    ckpt_path = os.path.join(config.result_root, model_name_str, f"{model_name_str}_{suffix}.pt")

    if not os.path.exists(ckpt_path):
        if accelerator.is_main_process:
            print(f"WARNING: Checkpoint not found at {ckpt_path}")
            print(f"         Skip loading, will train from scratch.")
        return None

    if accelerator.is_main_process:
        print(f"Loading checkpoint from: {ckpt_path}")

    data = torch.load(ckpt_path, map_location=torch.device(model.device))

    if load_step:
        step = data.get("step", 0)
        if accelerator.is_main_process:
            print(f"  Resuming from step: {step}")

    # 打印 checkpoint 里保存的关键超参数
    if "hparams" in data and accelerator.is_main_process:
        hp = data["hparams"]
        print(f"  Checkpoint hparams: beta={hp.get('beta')}, target_bpp={hp.get('target_bpp')}, "
              f"final_beta={hp.get('final_beta', hp.get('beta'))}")

    if "current_beta" in data and accelerator.is_main_process:
        print(f"  Last current_beta: {data['current_beta']:.6f}")

    model_to_load = accelerator.unwrap_model(model)
    try:
        missing, unexpected = model_to_load.module.load_state_dict(data["model"], strict=False)
    except AttributeError:
        missing, unexpected = model_to_load.load_state_dict(data["model"], strict=False)

    if accelerator.is_main_process:
        if missing:
            print(f"  Missing keys: {len(missing)} (showing first 5: {missing[:5]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (showing first 5: {unexpected[:5]})")

    return data  # 返回 checkpoint，trainer 可以读里面的 current_beta

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def data_load(config):
    path = config.data_config['data_path']
    resize = tuple([128, 256])
    if config.data_config['dataset_name'] == 'KITTI':
        train_dataset = PairKitti(path=path, set_type='train', resize=resize)
        val_dataset = PairKitti(path=path, set_type='val', resize=resize)
    elif config.data_config['dataset_name'] == 'Cityscape':
        train_dataset = PairCityscape(path=path, set_type='train', resize=resize)
        val_dataset = PairCityscape(path=path, set_type='val', resize=resize)
    else:
        raise Exception("Dataset not found")

    batch_size = config.batch_size

    g = torch.Generator()
    if getattr(config, 'seed', None) is not None:
        g.manual_seed(config.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val_batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def model_components(config):
    denoise = Unet(
        dim=config.embed_dim,
        channels=config.data_config["img_channel"],
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=config.context_dim_mults,
        # ===== [新增] 控制 mid block attention =====
        use_full_attn_mid=getattr(config, 'use_full_attn_mid', True),
    )

    context = ResnetCompressor(
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=config.data_config["img_channel"],
        out_channels=config.context_channels,
    )

    image_restore = None

    if config.compressor_ == 'big':
        context_w = ResnetCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=3,
            mode="bypass"
        )
        context_w_p = ResnetCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=3,
            mode="bypass"
        )
    elif config.compressor_ == 'f':
        context_w = f_w_compressor()
        context_w_p = f_w_compressor()

    return denoise, context, context_w, context_w_p, image_restore

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 固定随机种子
    if getattr(config, 'seed', None) is not None:
        set_seed(config.seed)
        if accelerator.is_main_process:
            print(f"Random seed fixed to {config.seed}")
    print(f"---------Target_bpp: {config.target_bpp} ----------")

    train_loader, val_loader = data_load(config)
    denoise, context, context_w, context_w_p, image_restore = model_components(config)
    print(model_name)

    model = DICM(
        device=args.device,
        denoise_fn=denoise,
        context_fn=context,
        context_w=context_w,
        context_w_p=context_w_p,
        clip_noise=config.clip_noise,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        vbr=config.vbr,
        lagrangian=config.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=config.alpha,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule,
        unroll_steps=config.unroll_steps,
        detach_unroll=config.detach_unroll,
        # ===== [新增] Rate Warmup =====
        rate_warmup_epochs=getattr(config, 'rate_warmup_epochs', 10),
        # AID 消融默认关闭；外部 config 未提供该字段时自动保持完整模型。
        aid_ablation=getattr(config, 'aid_ablation', False),
    ).to(args.device)
    step_start_ema=2000
    if config.load_model:
        step_start_ema=50
    trainer = Trainer(
        rank=args.device,
        accelerator=accelerator,
        sample_steps=config.sample_steps,
        diffusion_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        step_start_ema=step_start_ema,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        sample_mode=config.sample_mode,
        lagrangian=config.beta,
        # ===== [新增] 判别器独立学习率 =====
        disc_lr=getattr(config, 'disc_lr', 2.5e-5),
    )

    if config.load_model:
        load_name = getattr(config, 'load_model_name', model_name)
        load_suffix = getattr(config, 'load_suffix', 'best')

        ckpt_data = load(model, load_name, suffix=load_suffix, load_step=config.load_step)

        # 如果 checkpoint 里有 current_beta，让 trainer 接续之前的 β 状态
        if ckpt_data is not None and 'current_beta' in ckpt_data:
            trainer.current_beta = ckpt_data['current_beta']
            accelerator.unwrap_model(model).update_beta(ckpt_data['current_beta'])
            if accelerator.is_main_process:
                print(f"  Resumed current_beta = {trainer.current_beta:.6f}")
        if 'ema' in ckpt_data:
            trainer._init_ema_model()  # 先初始化 EMA model（拷贝当前主模型）
            trainer.ema_model.load_state_dict(ckpt_data['ema'], strict=False)
            if accelerator.is_main_process:
                print(f"  Resumed EMA weights")
    if config.load_diffusion:
        load_and_freeze_denoise_fn(model, "kitti/CDC_x.pt", config.freeze)
        new_layers = load_context_fn_partial_verbose(model, "kitti/CDC_x.pt")

    trainer.train()


if __name__ == "__main__":
    main()
