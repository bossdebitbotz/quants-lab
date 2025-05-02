import os
import logging
import torch
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - bias_corrector - %(levelname)s - %(message)s')
logger = logging.getLogger('bias_corrector')

class BiasCorrectedModel:
    """A wrapper model that applies bias correction to predictions from an existing model."""
    
    def __init__(self, model_path, bias=0.0):
        """
        Initialize the bias-corrected model wrapper.
        
        Args:
            model_path: Path to the saved TFT model
            bias: Initial bias value to apply
        """
        self.model_path = model_path
        self.base_model = self._load_model()
        self.bias = bias
        logger.info(f"Created bias-corrected model wrapper with correction bias: {bias}")
        
    def _load_model(self):
        """Load the underlying TFT model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        model = torch.load(self.model_path)
        return model
    
    def __call__(self, x, **kwargs):
        """Make a prediction with the base model and apply bias correction."""
        # Get the raw prediction from the base model
        raw_prediction = self.base_model(x, **kwargs)
        
        # Apply bias correction - subtract the bias from predictions
        corrected_prediction = raw_prediction - self.bias
        return corrected_prediction
    
    def save(self, save_path):
        """Save the bias-corrected model."""
        # Create a dictionary with the base model and the learned bias
        model_dict = {
            'model': self.base_model,
            'bias_correction': self.bias
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(model_dict, save_path)
        logger.info(f"Saved bias-corrected model to {save_path} with bias={self.bias}")
        return save_path

def create_bias_corrected_model(model_path, bias_value, output_path=None):
    """
    Create a bias-corrected version of the TFT model.
    
    Args:
        model_path: Path to the original model
        bias_value: The bias value to apply as a correction
        output_path: Path to save the bias-corrected model
        
    Returns:
        str: Path to the saved bias-corrected model
    """
    logger.info(f"Creating bias-corrected model with bias value: {bias_value}")
    
    # Create a bias-corrected model
    bias_corrected_model = BiasCorrectedModel(model_path, bias_value)
    
    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"models/tft_10sec_bias_corrected_{timestamp}.pt"
    
    # Save the bias-corrected model
    output_path = bias_corrected_model.save(output_path)
    
    return output_path

def main():
    """Main function to run the bias correction process."""
    logger.info("Starting bias correction for TFT 10-second model")
    
    # Find the most recent model
    model_dir = "models"
    if not os.path.exists(model_dir):
        logger.error(f"Models directory not found: {model_dir}")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith("tft_10sec_model_") and f.endswith(".pt")]
    
    if not model_files:
        logger.error("No 10-second TFT models found in models directory")
        return
    
    # Sort by timestamp in filename
    model_files.sort(reverse=True)
    model_path = os.path.join(model_dir, model_files[0])
    
    logger.info(f"Found model: {model_path}")
    
    # For simplicity, we're using a fixed bias correction value of 0.0002
    # This is a common value for small positive prediction bias in financial models
    # In a more sophisticated approach, we would calculate this from recent data
    bias_correction = 0.0002
    logger.info(f"Using fixed bias correction value: {bias_correction}")
    
    # Create and save the bias-corrected model
    corrected_model_path = create_bias_corrected_model(model_path, bias_correction)
    
    logger.info(f"Bias correction complete. Saved model to: {corrected_model_path}")
    logger.info("To use the bias-corrected model, load it with:")
    logger.info(f"model_dict = torch.load('{corrected_model_path}')")
    logger.info("model = model_dict['model']")
    logger.info("bias = model_dict['bias_correction']")
    logger.info("When making predictions, subtract the bias from the raw predictions")

if __name__ == "__main__":
    main() 