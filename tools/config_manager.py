import json
from pathlib import Path

class ConfigManager:
    def __init__(self,config_path=None):
        self.config = {
            'vocab_size': 50000,
            'device': 'cpu',
            'n_layer': 12,
            'pad_idx': 0,
            'n_head': 12,
            'd_model': 768,
            'd_c': 64,
            'd_r': 64,
            'hidden': 3072,
            'other_experts': 8,
            'shared_experts': 2,
            'keep': 4,
            'ro_theta': 10000.0,
            'dropout': 0.1,
            'scale': 0.02,
            'alpha': 0.01
        }
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        self.config.update(loaded_config)