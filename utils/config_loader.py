import yaml
from typing import Dict, Any


class ConfigLoader:
    """Load and merge YAML configuration files"""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load a YAML config file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save(config: Dict[str, Any], output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def merge(base_config: Dict, override_config: Dict) -> Dict:
        """Recursively merge override_config into base_config"""
        result = base_config.copy()
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge(result[key], value)
            else:
                result[key] = value
        return result
