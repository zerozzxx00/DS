import json
import logging
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config_path = Path.home() / ".study_assistant" / "config.json"
        self.config = self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"加载配置失败: {str(e)}")
            return {}

    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)

    def save_config(self, new_config):
        """保存配置更新"""
        try:
            self.config.update(new_config)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"保存配置失败: {str(e)}")