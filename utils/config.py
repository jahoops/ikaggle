from pathlib import Path

import yaml


class Config:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, section, option, default=None):
        return self.config.get(section, {}).get(option, default)

# Singleton instance
config = Config()