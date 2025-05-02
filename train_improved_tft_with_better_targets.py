import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from models.improved_tft import TemporalFusionTransformer
from sklearn.preprocessing import StandardScaler
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_better_targets")

# Custom Dataset class to work with prepared data
class ImprovedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data: pd.DataFrame,
        feature_cols,
        target_col='target_sign',  # Default to sign-based target
        context_length=10,
        prediction_length=1
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Calculate valid samples (need enough history for context)
        self.num_samples = max(0, len(data) - context_length - prediction_length + 1)
        
        logger.info(f"Created dataset with {self.num_samples} samples")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Extract sequence window
        history_window = slice(idx, idx + self.context_length)
        future_window = slice(idx + self.context_length, idx + self.context_length + self.prediction_length)
        
        # Extract features from history window
        features = self.data.iloc[history_window][self.feature_cols].values
        features = torch.tensor(features, dtype=torch.float32)
        
        # Extract target from future window
        target = self.data.iloc[future_window][self.target_col].values
        target = torch.tensor(target, dtype=torch.float32)
        
        return {
            'temporal_real_features': features,
            'targets': target
        }

# Custom Loss Function for Direction Accuracy
class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        # Alpha controls the balance between MSE and direction loss
        # alpha=0: pure MSE, alpha=1: pure direction
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # Mean squared error component
        mse_loss = self.mse(y_pred, y_true)
        
        # Direction accuracy component (converted to loss)
        # Get signs of predictions and targets
        pred_sign = torch.sign(y_pred)
        true_sign = torch.sign(y_true)
        
        # Calculate sign match as a percentage (1 = all match, 0 = none match)
        sign_match = (pred_sign == true_sign).float().mean()
        
        # Convert to loss (1 - match)
        direction_loss = 1.0 - sign_match
        
        # Combine losses
        combined_loss = (1 - self.alpha) * mse_loss + self.alpha * direction_loss
        
        return combined_loss

class DirectionalTFT(TemporalFusionTransformer):
    """Extended TFT with focus on directional accuracy"""
    
    def __init__(self, *args, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.directional_loss = DirectionalLoss(alpha=alpha)
        
        # Add tracking for direction accuracy
        self.train_acc = []
        self.val_acc = []
    
    def _apply_loss(self, y_pred, y_true):
        if self.loss_fn == "directional":
            return self.directional_loss(y_pred, y_true)
        else:
            return super()._apply_loss(y_pred, y_true)
    
    def training_step(self, batch, batch_idx):
        # Extract data from batch
        static_categorical = batch.get('static_categorical_features', None)
        static_real = batch.get('static_real_features', None)
        temporal_categorical = batch.get('temporal_categorical_features', None)
        temporal_real = batch['temporal_real_features']
        targets = batch['targets']
        
        # Forward pass
        outputs = self.forward(
            static_categorical_features=static_categorical,
            static_real_features=static_real, 
            temporal_categorical_features=temporal_categorical,
            temporal_real_features=temporal_real
        )
        
        predictions = outputs['prediction']
        
        # Get predictions for target timepoints
        if self.loss_fn in ["mse", "directional"]:
            predictions = predictions.squeeze(-1)
        
        # Calculate loss
        loss = self._apply_loss(predictions, targets)
        
        # Calculate direction accuracy
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(targets)
        direction_acc = (pred_sign == true_sign).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_dir_acc', direction_acc, prog_bar=True)
        
        # Store for later analysis
        self.train_acc.append(direction_acc.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Same as training, but for validation
        static_categorical = batch.get('static_categorical_features', None)
        static_real = batch.get('static_real_features', None)
        temporal_categorical = batch.get('temporal_categorical_features', None)
        temporal_real = batch['temporal_real_features']
        targets = batch['targets']
        
        outputs = self.forward(
            static_categorical_features=static_categorical,
            static_real_features=static_real, 
            temporal_categorical_features=temporal_categorical,
            temporal_real_features=temporal_real
        )
        
        predictions = outputs['prediction']
        if self.loss_fn in ["mse", "directional"]:
            predictions = predictions.squeeze(-1)
        
        loss = self._apply_loss(predictions, targets)
        
        # Direction accuracy
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(targets)
        direction_acc = (pred_sign == true_sign).float().mean()
        
        # Calculate additional metrics
        mae = torch.abs(predictions - targets).mean()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dir_acc', direction_acc, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)
        
        # Store for later analysis
        self.val_acc.append(direction_acc.item())
        
        return loss
    
    def configure_optimizers(self):
        # Use the same optimizer as base TFT
        return super().configure_optimizers()

def load_data(file_path="data_improvements/improved_dataset_small.csv"):
    """Load the improved dataset"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def prepare_data(df, target_col='target_sign', context_length=10, prediction_length=1):
    """Prepare data for training"""
    # Define feature columns to use
    # Exclude target columns and timestamp
    target_columns = ['original_target', 'target_sign', 'target_binned', 'target_log', 'target_ma_adjusted']
    feature_cols = [col for col in df.columns if col not in target_columns]
    
    # Normalize features
    scaled_df = df.copy()
    scalers = {}
    
    for col in feature_cols:
        if col in ['hour', 'minute']:  # Categorical features, don't scale
            continue
            
        # Handle any remaining NaN or infinite values
        if scaled_df[col].isnull().any() or np.isinf(scaled_df[col]).any():
            logger.warning(f"Column {col} contains NaN or inf values. Replacing with 0.")
            scaled_df[col] = scaled_df[col].replace([np.inf, -np.inf, np.nan], 0)
        
        # Scale the feature
        scaler = StandardScaler()
        scaled_df[col] = scaler.fit_transform(scaled_df[[col]])
        scalers[col] = scaler
    
    # Create train/val/test splits
    train_size = int(len(scaled_df) * 0.7)
    val_size = int(len(scaled_df) * 0.15)
    
    train_data = scaled_df.iloc[:train_size]
    val_data = scaled_df.iloc[train_size:train_size+val_size]
    test_data = scaled_df.iloc[train_size+val_size:]
    
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    
    # Create datasets
    train_dataset = ImprovedDataset(
        train_data, 
        feature_cols=feature_cols, 
        target_col=target_col,
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    val_dataset = ImprovedDataset(
        val_data, 
        feature_cols=feature_cols, 
        target_col=target_col,
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    test_dataset = ImprovedDataset(
        test_data, 
        feature_cols=feature_cols, 
        target_col=target_col,
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    return train_dataset, val_dataset, test_dataset, feature_cols, scalers

def train_directional_tft(
    train_dataset,
    val_dataset,
    test_dataset,
    num_features,
    hidden_size=64,
    num_attention_heads=2,
    dropout=0.2,
    alpha=0.5,
    learning_rate=5e-4,
    batch_size=16,
    max_epochs=50,
    patience=10,
    context_length=10,
    use_gpu=True
):
    """Train the TFT model with directional loss"""
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = DirectionalTFT(
        static_variables=[],  # No static variables
        time_varying_categorical_variables=[],  # No categorical variables for now
        time_varying_real_variables=num_features,
        static_embedding_sizes=[],  # No static embeddings
        time_varying_embedding_sizes=[],  # No categorical embeddings
        hidden_size=hidden_size,
        lstm_layers=2,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        context_length=context_length,
        prediction_length=1,
        loss_fn="directional",  # Use our custom directional loss
        alpha=alpha,  # Balance between MSE and direction loss
        bias_correction=True
    )
    
    # Create checkpoint callback
    checkpoint_dir = f"checkpoints/directional_tft_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tft-{epoch:02d}-{val_dir_acc:.4f}',  # Monitor directional accuracy
        save_top_k=3,
        monitor='val_dir_acc',
        mode='max'  # We want to maximize accuracy
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_dir_acc',
        patience=patience,
        mode='max'
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up logger
    logger_dir = f"lightning_logs/directional_tft_{timestamp}"
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    # Try to set up TensorBoard logger
    try:
        tb_logger = TensorBoardLogger(save_dir=logger_dir)
        logger_to_use = tb_logger
    except (ModuleNotFoundError, ImportError):
        logger.warning("TensorBoard not available, using CSVLogger instead")
        from pytorch_lightning.loggers import CSVLogger
        csv_logger = CSVLogger(save_dir=logger_dir)
        logger_to_use = csv_logger
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger_to_use,
        gradient_clip_val=1.0,  # Prevent gradient explosion
        log_every_n_steps=10
    )
    
    # Train model
    logger.info(f"Starting training with {len(train_loader.dataset)} samples")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    logger.info(f"Evaluating model on test set with {len(test_loader.dataset)} samples")
    test_results = trainer.test(model, dataloaders=test_loader)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/directional_tft_{timestamp}.pt"
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'directional_tft',
        'hyperparameters': {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'dropout': dropout,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'context_length': context_length
        },
        'num_features': num_features,
        'test_results': test_results[0] if test_results else None,
        'best_val_acc': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        'best_epoch': checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0] if checkpoint_callback.best_model_path else None
    }
    
    # Load best model if available
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        best_model = DirectionalTFT.load_from_checkpoint(best_model_path)
        
        # Save model package
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'metadata': metadata,
            'config': best_model.hparams
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return best_model, metadata
    else:
        logger.warning("No checkpoint found, saving last model state")
        
        # Save current model
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'config': model.hparams
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model, metadata

def main():
    parser = argparse.ArgumentParser(description='Train TFT model with better targets')
    parser.add_argument('--target', type=str, default='target_sign', 
                        choices=['target_sign', 'target_binned', 'target_log', 'target_ma_adjusted', 'original_target'],
                        help='Target column to use for training')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Balance between MSE (0) and directional accuracy (1)')
    parser.add_argument('--context', type=int, default=10, help='Context length (time steps)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden size')
    parser.add_argument('--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training even if GPU is available')
    parser.add_argument('--dataset', type=str, default='data_improvements/improved_dataset_small.csv',
                       help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.dataset)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, feature_cols, scalers = prepare_data(
        df, 
        target_col=args.target,
        context_length=args.context,
        prediction_length=1
    )
    
    # Train model
    model, metadata = train_directional_tft(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_features=len(feature_cols),
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        dropout=args.dropout,
        alpha=args.alpha,
        learning_rate=args.lr,
        batch_size=args.batch,
        max_epochs=args.epochs,
        patience=args.patience,
        context_length=args.context,
        use_gpu=not args.cpu
    )
    
    logger.info("Training completed!")
    logger.info(f"Test results: {metadata['test_results']}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.train_acc, label='Train')
    plt.plot(model.val_acc, label='Validation')
    plt.title('Directional Accuracy During Training')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/directional_training_curves_{metadata['timestamp']}.png")
    logger.info(f"Training curves saved to {plots_dir}/directional_training_curves_{metadata['timestamp']}.png")

if __name__ == "__main__":
    main() 