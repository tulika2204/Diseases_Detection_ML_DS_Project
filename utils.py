import yaml

def load_config(config_path='./config.yaml'):
    """Load configuration settings from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}
