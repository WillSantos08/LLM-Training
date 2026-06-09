import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        possible_paths = [
            'config.yaml',
            '../config.yaml',
            '../../config.yaml',
            'cephalon_luna/../config.yaml',
        ]
        
        config_path = None
        for p in possible_paths:
            if Path(p).exists():
                config_path = Path(p)
                break
        
        if config_path is None:
            print("⚠ Configuration file not found")
            print("   Searched in:")
            for p in possible_paths:
                print(f"     - {p}")
            print("   Using default configuration...")
            return {}
    else:
        config_path = Path(path)
        if not config_path.exists():
            print(f"⚠ Configuration file not found: {path}")
            print("   Using default configuration...")
            return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded from: {config_path.absolute()}")
        
        return config if config else {}
    
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return {}