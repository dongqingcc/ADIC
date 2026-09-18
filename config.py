# training config
n_step = 1000000
scheduler_checkpoint_step = 500
log_checkpoint_step = 50
gradient_accumulation_steps=1

lr = 0.00005
batch_size =14  #  u1: f-21  big-18  u2: f-16
val_batch_size = 14 # 64

SWITCH_EPOCH = 5000
train_sample_steps=15

decay = 0.9
minf = 0.5
optimizer = "adamw"  # adamw or adam
n_workers = 12

# load
load_model = False
load_step = load_model
load_diffusion=False
freeze=4

load_model_name="l1-KITTI-t8000-v500-t0.05-b0.065-vbrFalse-f_compressor-u_s2_noise_add-x-cosine-aux0.8lpips-seed113333_s113"


# diffusion config
alpha=0.8
beta=0.065

target_bpp = 0.18          # 目标 BPP（None 表示禁用自适应，使用固定 β）
 
beta_lr = 0.2            # 对数空间学习率
                          # 当前值含义：ratio=2 时 β *= exp(0.05*log(2)) ≈ 1.035 (涨3.5%)
                          # 配合每 100 step 才更新，β 整体调整很慢很稳
 
beta_min = 0.001          # β 下限（极低质量保护）
beta_max = 2.0            # β 上限（极高码率压缩保护）
 

 
# ===== Rate Warmup =====
rate_warmup_epochs = 50   # 前 50 epoch 用固定 β（不开自适应），让模型先收敛
 
# ===== [新增] BPP 平滑参数 =====
bpp_ema_decay = 0.99      # BPP 滑动平均衰减率（约 100 step 窗口）
                          # 越接近 1 越平滑但反应慢
                          # 0.99 表示新观测占 1%，旧均值占 99%
aid_ablation = True 
# ===== [新增] β 更新周期 =====
beta_update_freq = 100    # 每 100 个 step 才更新一次 β
                          # 让模型有时间响应上一次 β 的变化
                          # 避免 P 控制器震荡（β 改了但模型还没适应就再改）
disc_lr = 2.5e-5
use_full_attn_mid = True
seed = 113  # 或任意你想指定的值，设为 None 则不固定

relay_start_t: 7000
init="noise" # noise or compress
restore_loss_weight=1
fusion="add" #add or f
pred_mode = 'x'
loss_type = "l1"
iteration_step = 8000                   #20000
sample_steps = 500     #500
embed_dim = 64    #64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "none"
val_num_of_batch = 1
additional_note = "xiaorong_s113"
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "cosine"
aux_loss_type = "lpips"
compressor_ = "f"  # big or f
detach_unroll=False
unroll_steps=2
# data config
data_config = {
    "dataset_name": "KITTI",
    "data_path": "/share/jobdata/d1049732203093/hq/dataset/data",
    "sequence_length": 1,
    "img_size": 256,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}



result_root = "./kitti"
tensorboard_root = "*"
