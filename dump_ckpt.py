import torch
import os
from tqdm import tqdm
import yaml

ckpt_root = "/fsx-project/xueyao/ckpt/repcodec/hubert_large_l18"

for c in [1024, 2048, 4096, 8192, 16384]:
    print("For", c)

    ckpt_dir = os.path.join(ckpt_root, f"c{c}")
    ckpt_file = os.path.join(ckpt_dir, "checkpoint-200000steps.pkl")
    config_file = os.path.join(ckpt_dir, "config.yml")

    output_ckpt_file = os.path.join(ckpt_dir, f"hubert_large_l18_c{c}.pkl")
    output_config_file = os.path.join(ckpt_dir, f"hubert_large_l18_c{c}.yaml")

    with open(config_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    model_config = data["model_params"]
    with open(output_config_file, 'w', encoding='utf-8') as file:
        yaml.dump(model_config, file, default_flow_style=False, allow_unicode=True)

    cpu_weights = torch.load(ckpt_file, map_location="cpu")
    cpu_weights = {k:v for k, v in cpu_weights.items() if k == "model"}
    torch.save(cpu_weights, output_ckpt_file)
