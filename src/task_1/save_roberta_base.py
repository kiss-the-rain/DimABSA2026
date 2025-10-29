# save_roberta_base.py（在有网的电脑上跑）
from transformers import AutoTokenizer, AutoModel
local_dir = "models/roberta-base"
tok = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
mdl = AutoModel.from_pretrained("roberta-base")
tok.save_pretrained(local_dir)
mdl.save_pretrained(local_dir)
print("Saved:", local_dir)
