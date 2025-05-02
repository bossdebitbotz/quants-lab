import torch
import numpy as np
import pandas as pd
import logging

from train_improved_tft_with_better_targets import load_data, DirectionalTFT
from train_downward_specialist_tft import DownwardSpecialistTFT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_inspector")

def load_model(model_path, model_type="directional"):
    """Load a trained model from file"""
    if not model_path or not model_type:
        return None, None
        
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        config = checkpoint.get('config', None)
        
        logger.info(f"Model config from {model_path}: {config}")
        
        return checkpoint, config
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None, None

def inspect_dataset_features(dataset_path='data_improvements/improved_dataset_small.csv'):
    """Inspect the dataset's feature dimensions"""
    df = load_data(dataset_path)
    if df is None:
        logger.error("Failed to load dataset")
        return
    
    # Select features (all columns except targets and timestamp)
    feature_cols = [col for col in df.columns if not col.startswith('target_') 
                   and col != 'timestamp' and col != 'original_target']
    
    logger.info(f"Dataset has {len(feature_cols)} features: {feature_cols}")
    
    # Count all columns by type
    target_cols = [col for col in df.columns if col.startswith('target_') or col == 'original_target']
    logger.info(f"Target columns ({len(target_cols)}): {target_cols}")
    
    # Print sample data shape
    logger.info(f"Dataset shape: {df.shape}")
    
    return df, feature_cols

def extract_tensors_from_model(checkpoint):
    """Extract key tensors from the model checkpoint to inspect dimensions"""
    if not checkpoint:
        return
        
    state_dict = checkpoint.get('model_state_dict', {})
    key_layers = {}
    
    # Find input/output layers
    for key, tensor in state_dict.items():
        if 'weight' in key and any(x in key for x in ['transform', 'embedding', 'layer.weight']):
            key_layers[key] = tensor.shape
    
    logger.info("Key layer dimensions:")
    for name, shape in key_layers.items():
        logger.info(f"  {name}: {shape}")

def main():
    # Inspect directional model
    logger.info("Inspecting directional model...")
    dir_checkpoint, dir_config = load_model('models/directional_tft_20250502_134913.pt', 'directional')
    
    # Inspect downward model
    logger.info("\nInspecting downward specialist model...")
    down_checkpoint, down_config = load_model('models/downward_specialist_tft_20250502_140300.pt', 'downward')
    
    # Extract tensor dimensions
    logger.info("\nDirectional model tensors:")
    extract_tensors_from_model(dir_checkpoint)
    
    logger.info("\nDownward specialist model tensors:")
    extract_tensors_from_model(down_checkpoint)
    
    # Inspect dataset
    logger.info("\nInspecting dataset features:")
    df, feature_cols = inspect_dataset_features()
    
    # Check if feature count matches the model's expected input size
    if dir_config and 'time_varying_real_variables' in dir_config:
        logger.info(f"\nDirectional model expects {dir_config['time_varying_real_variables']} features")
        if len(feature_cols) != dir_config['time_varying_real_variables']:
            logger.error(f"Mismatch between dataset features ({len(feature_cols)}) and directional model input size ({dir_config['time_varying_real_variables']})")
    
    if down_config and 'time_varying_real_variables' in down_config:
        logger.info(f"Downward model expects {down_config['time_varying_real_variables']} features")
        if len(feature_cols) != down_config['time_varying_real_variables']:
            logger.error(f"Mismatch between dataset features ({len(feature_cols)}) and downward model input size ({down_config['time_varying_real_variables']})")

if __name__ == "__main__":
    main() 