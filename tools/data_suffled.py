import random
import numpy as np
import torch

def data_shuffled(json_data_path, new_data_path, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open(json_data_path, "r", encoding='utf-8') as data_file:
        lines = data_file.readlines()
    random.shuffle(lines)
    with open(new_data_path, "w", encoding='utf-8') as new_file:
        new_file.writelines(lines)

if __name__ == '__main__':    
    data_shuffled(
        json_data_path="dataset\TinySeek_dataset\mix_gated.jsonl",
        new_data_path="dataset\TinySeek_dataset\mix_gated_shuffled_1.jsonl"
    )
    data_shuffled(
        json_data_path="dataset\TinySeek_dataset\mix_gated_shuffled_1.jsonl",
        new_data_path="dataset\TinySeek_dataset\mix_gated_shuffled.jsonl"
    )