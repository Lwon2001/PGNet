import os
import yaml
import time
import shutil
from datetime import datetime


def load_config(config_path):
    """
    Load configuration from a YAML file with support for imports
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle imports
    if '_import' in config:
        base_config_path = config['_import']
        # Remove the import key from config
        del config['_import']
        
        # Load the base config
        base_config = load_config(base_config_path)
        
        # Merge configs - base config is overridden by specific config
        merged_config = deep_update(base_config, config)
        return merged_config
    
    return config


def deep_update(base_dict, update_dict):
    """
    Recursively update a nested dictionary without removing keys from base_dict
    that don't exist in update_dict
    
    Args:
        base_dict (dict): The base dictionary to update
        update_dict (dict): The dictionary with updated values
        
    Returns:
        dict: Updated dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # If both values are dicts, recursively update
            result[key] = deep_update(result[key], value)
        else:
            # Otherwise, just update the value
            result[key] = value
            
    return result


def save_config(config, save_path):
    """
    Save configuration to a YAML file
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path where to save the configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_output_dirs(config, config_for_saving=None):
    """
    Set up output directories based on configuration
    
    Args:
        config (dict): Configuration dictionary for path creation
        config_for_saving (dict, optional): Config to save in the experiment directory.
                                           If None, the input config will be saved.
        
    Returns:
        dict: Dictionary with paths for logs, checkpoints, and configs
    """
    # Extract parameters from config
    category = config['dataset']['category']
    model_name = config['model']['name']
    sampling_method = config['dataset']['sampling_method']
    num_points = config['dataset']['gen_points']
    
    # Create base timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base directories
    base_path = config['output']['base_path']
    # 确保base_path是一个绝对路径，如果是相对路径则相对于项目根目录
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), base_path)
    
    # Create experiment name
    experiment_name = f"{model_name}_{category}_{sampling_method}_{num_points}_{timestamp}"
    
    # Create directories
    paths = {
        'base': os.path.join(base_path, experiment_name),
        'logs': os.path.join(base_path, experiment_name, config['output']['log_dir']),
        'checkpoints': os.path.join(base_path, experiment_name, config['output']['checkpoint_dir']),
        'configs': os.path.join(base_path, experiment_name, config['output']['config_dir']),
        'experiment_name': experiment_name,
        'timestamp': timestamp
    }
    
    # Create directories if they don't exist
    for key, path in paths.items():
        # Skip experiment_name and timestamp as they're not directories
        if key not in ['experiment_name', 'timestamp'] and isinstance(path, str) and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
    # Save config for reproducibility
    config_save_path = os.path.join(paths['configs'], f"config_{timestamp}.yaml")
    # Use config_for_saving if provided, otherwise use the input config
    save_config(config_for_saving if config_for_saving is not None else config, config_save_path)
    
    return paths 