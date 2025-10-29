# model.py
from dataclasses import dataclass
from pathlib import Path
import torch, random, numpy as np
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict, defaultdict
from src.utils.paths import output_path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

 # -------------------- 基础配置 --------------------
@dataclass
# 存储参数
class Config:
    model_name: str = "roberta-base"   # 中文可换 hfl/chinese-roberta-wwm-ext
    max_len: int = 192
    lr_encoder: float = 1e-5
    lr_head: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 5
    batch_size: int = 16
    grad_accum: int = 1
    seed: int = 42


# 实用函数
def set_seed(s):
    random.seed(s); 
    np.random.seed(s); 
    torch.manual_seed(s)

def get_device():
    # 优先 MPS (Apple GPU)，再 CUDA，最后 CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
device = get_device()
torch.set_float32_matmul_precision("high")

tok = AutoTokenizer.from_pretrained(Config.model_name, use_fast=True) # 选择分词器

# -------------------- 数据处理 --------------------
# 传入文本，通过分词器转换为向量
def build_inputs(text: str, aspect: str, max_len: int):
    prompt = f"Aspect: {aspect}"
    return tok(
    text, prompt,
    truncation="only_first",
    padding="max_length",
    max_length=max_len,
    return_tensors="pt"
)

# 训练集数据处理
class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True); 
        self.max_len = max_len
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i] # 取df的第i行
        # 取数据
        enc = build_inputs(r["text"], r["aspect"], self.max_len)
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),       # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [L]
            "labels": torch.tensor(
                [float(r["v"]), float(r["a"])], dtype=torch.float32
            ),  # [2]
        }
        # 仅当存在时才添加，避免 DataLoader collate 遇到 NoneType
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)  # [L]
        return item
# 测试集数据处理
class DevDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df.reset_index(drop=True); self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = build_inputs(r["text"], r["aspect"], self.max_len)
        item = {
            "id":      str(r["id"]),     # 保持为可序列化的标量
            "aspect":  r["aspect"],      # 原样回写用于提交
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item
# -------------------- 模型构建 --------------------
class VARegressor(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name) # 加载编码器
        hidden = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 2)  # 输出 [V, A]
        )
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]        # [B, H] 
        # 从编码器输出的所有 token 隐状态中，取每个序列的“第 0 个 token 向量” 即CLS向量（整个句子的向量表示）
        logits = self.head(cls)                  # [B, 2]
        return logits
# -------------------- 损失与指标 --------------------
def loss_fn(pred, target, huber=False): # 定义损失函数
    if huber:
        return F.smooth_l1_loss(pred, target)   # Huber
    return F.mse_loss(pred, target) # 均方误差

def rmse_va(pred, target): # 计算的是整体 RMSE：把 V 与 A 两个维度、以及 整个 batch 的误差一起平均，再开根号。
    # pred/target: [B,2]  →  单步 RMSE（训练日志用）
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse)

# 定义一个辅助函数，同时构建优化器和学习率调度器
def build_optim_sched(model, num_train_steps, lr_enc, lr_head, wd, warmup_ratio):
    '''
    model:  你的 nn.Module（里有 self.enc 与回归头）。
    num_train_steps:  总训练步数（= 所有 epoch 的 len(dataloader) 之和）。
    lr_enc / lr_head:  编码器与头部的分层学习率。
    wd:  weight_decay（权重衰减/L2）。
    warmup_ratio:  预热比例（例如 0.1 表示前 10% 步数线性升到基准 lr）。
    '''
    enc_params, head_params = [], []
    for n, p in model.named_parameters(): # n表示parameters的名字，p表示对应的参数对象
        if "enc." in n:  # 名字包含 "enc." 来判断是否属于编码器
            enc_params.append(p)
        else: 
            head_params.append(p)
    optimizer = torch.optim.AdamW([ # 构建AdamW优化器
        {"params": enc_params, "lr": lr_enc, "weight_decay": wd}, # 编码器优化器
        {"params": head_params, "lr": lr_head, "weight_decay": wd}, # 头部优化器
    ])
    num_warmup = int(num_train_steps * warmup_ratio) # 把 warmup 比例转为预热步数。
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)  
    # 使用 HuggingFace Transformers 的线性预热 + 线性下降调度器：
    return optimizer, scheduler  # 返回优化器和调度器

def train_one_epoch(model, loader, optimizer, scheduler, device, use_amp=True, huber=False, max_grad_norm=1.0):
    '''
    model：你的 VARegressor
    loader：训练 DataLoader
    optimizer / scheduler：优化器与学习率调度器
    device：cpu/cuda/mps
    use_amp：是否启用自动混合精度
    huber：损失是否用 Huber（SmoothL1）
    max_grad_norm：梯度裁剪阈值
    '''
    model.train()
    scaler = GradScaler(enabled=use_amp) 
    # AMP 的梯度缩放器；开启后会动态放大/缩小 loss，避免半精度下的 underflow（数值变 0）。
    total_loss, total_rmse, n = 0.0, 0.0, 0
    for batch in loader: 
        # 开始提取数据
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels = batch["labels"].to(device)
        # 开始训练
        optimizer.zero_grad(set_to_none=True)
        # autocast 会在合适算子上用 FP16/BF16，提高吞吐、节省显存。
        amp_dtype = torch.float16  # MPS/ CUDA 下都可先用 FP16；CPU 可改 bfloat16
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            pred = model(input_ids, attn, ttids)
            loss = loss_fn(pred, labels, huber=huber)
        scaler.scale(loss).backward() # 反向传播 对 loss 缩放后反传。此处不会立刻更新参数，只是把梯度累在 .grad 上。
        #?等价于对 loss * scale 反传，所有梯度都会被乘以 scale，下溢风险显著降低。
        #?之后更新参数前需要把这层“放大”去掉，所以要 unscale_。
        scaler.unscale_(optimizer)    # 把之前缩放过的梯度还原回真实尺度；为了下一步能正确做梯度裁剪或检查梯度值。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
        # 对所有参数的梯度做 L2 范数裁剪，限制到 max_grad_norm。防止梯度爆炸，稳定训练。
        scaler.step(optimizer)
        scaler.update()
        # scaler.step(optimizer)：用缩放安全的方式执行 optimizer.step()（内部会根据梯度是否为 NaN/Inf 决定跳过或执行）。
        scheduler.step() # 更新参数

        # 累计加权和与样本数
        total_loss += loss.item() * len(input_ids)
        total_rmse += rmse_va(pred.detach(), labels).item() * len(input_ids)
        n += len(input_ids)
    return total_loss/n, total_rmse/n  # 返回按样本数加权的 epoch 平均损失/平均 RMSE。

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_rmse, n = 0.0, 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        labels = batch["labels"].to(device)

        pred = model(input_ids, attn, ttids)
        loss = loss_fn(pred, labels)
        total_loss += loss.item() * len(input_ids)
        total_rmse += rmse_va(pred, labels).item() * len(input_ids)
        n += len(input_ids)
    return total_loss/n, total_rmse/n


def fit(train_df, cfg: Config):
    set_seed(cfg.seed)
    tr_df, va_df = train_test_split(train_df, test_size=0.1, random_state=cfg.seed)# 拆分训练和验证集
    tr_ds = TrainDataset(tr_df, cfg.max_len)
    va_ds = TrainDataset(va_df, cfg.max_len) # 将DataFrame转换为PyTorch Dataset
    
    # 构建DataLoader
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 实例化模型
    model = VARegressor(cfg.model_name).to(device)
    steps = len(tr_loader) * cfg.epochs
    optimizer, scheduler = build_optim_sched(model, steps, cfg.lr_encoder, cfg.lr_head, cfg.weight_decay, cfg.warmup_ratio)

    best_rmse, best_path = 1e9, "best_model.pt" # 初始化验证集 RMSE 的最好值与 checkpoint 路径
    for ep in range(cfg.epochs):
        tl, tr = train_one_epoch(model, tr_loader, optimizer, scheduler, device, use_amp=(device.type!="cpu"))
        vl, vr = evaluate(model, va_loader, device)
        print(f"Epoch {ep+1:02d} | train_loss={tl:.4f} rmse={tr:.3f} || val_loss={vl:.4f} rmse={vr:.3f}")
        if vr < best_rmse: # 保存最好的模型
            best_rmse = vr
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            print(f"[BEST] saved -> {best_path} (rmse={best_rmse:.3f})")

def format_va(v, a):
    v = float(max(1.0, min(9.0, v)))
    a = float(max(1.0, min(9.0, a)))
    return f"{v:.2f}#{a:.2f}"

@torch.no_grad()
def predict_dev(model, dev_df, cfg: Config):
    ds = DevDataset(dev_df, cfg.max_len) # 用你的 DevDataset 封装样本，负责把 (text, aspect) 编码成定长序列
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2) # 批量加载，不打乱顺序
    model.eval()
    records = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        ttids = batch.get("token_type_ids")
        ttids = ttids.to(device) if ttids is not None else None
        pred = model(input_ids, attn, ttids).cpu().numpy()  # [B,2]
        for i, (vid, asp) in enumerate(zip(batch["id"], batch["aspect"])):
            # vid 是样本 ID，asp 是对应的 Aspect
            v, a = pred[i].tolist()
            records.append((vid, asp, v, a))
    # 组装为官方 JSONL 所需结构：每个 ID 一行 + Aspect_VA 列表
    bag = defaultdict(list)
    for rid, asp, v, a in records:
        bag[rid].append({"Aspect": asp, "VA": format_va(v, a)})
        # 组装成 {"Aspect": 原样的 aspect, "VA": "V#A"} 放入该 ID 的列表中。
    lines = []
    for rid, items in bag.items():
        lines.append({"ID": rid, "Aspect_VA": items})
        # 最终输出为一个行对象列表：{"ID": "R001", "Aspect_VA": [{"Aspect":"thai food","VA":"6.75#6.38"}, ...]}
    return lines

def save_jsonl(objs, path):
    import json, os
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for o in objs:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")

def main():
    cfg = Config()
    # 读取 dataset.py 输出：
    train_df = pd.read_parquet(output_path("output","track_a","subtask_1") / "train_pairs.parquet")
    dev_df   = pd.read_parquet(output_path("output","track_a","subtask_1") / "dev_pairs.parquet")

    fit(train_df, cfg)

    ckpt = torch.load("best_model.pt", map_location="cpu")
    model = VARegressor(cfg.model_name); model.load_state_dict(ckpt["state_dict"]); model.to(device)

    preds = predict_dev(model, dev_df, cfg)
    out_file = output_path("submit","task1","pred_dev.jsonl")
    save_jsonl(preds, out_file)
    print("Dev 提交文件 ->", out_file)

if __name__ == "__main__":
    main()
