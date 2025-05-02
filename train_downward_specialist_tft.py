import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import matplotlib.pyplot as plt
from train_improved_tft_with_better_targets import (
    ImprovedDataset, DirectionalTFT, load_data, prepare_data
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("downward_tft")

# Custom Loss Function with higher penalty for wrong downward predictions
class DownwardFocusLoss(nn.Module):
    def __init__(self, alpha=0.7, downward_weight=3.0):
        super().__init__()
        # Alpha controls the balance between MSE and direction loss
        # Higher alpha (closer to 1) puts more weight on directional accuracy
        self.alpha = alpha
        # Weight for downward movements (negative signs)
        self.downward_weight = downward_weight
        self.mse = nn.MSELoss(reduction='none')  # Use 'none' to apply weights later
    
    def forward(self, y_pred, y_true):
        # Create weights - higher for negative target values
        weights = torch.ones_like(y_true)
        weights[y_true < 0] = self.downward_weight
        
        # Mean squared error component with weights
        mse_loss_raw = self.mse(y_pred, y_true)
        mse_loss = (mse_loss_raw * weights).mean()
        
        # Direction accuracy component (converted to loss)
        # Get signs of predictions and targets
        pred_sign = torch.sign(y_pred)
        true_sign = torch.sign(y_true)
        
        # Apply higher weight for downward mispredictions
        dir_weights = torch.ones_like(true_sign)
        dir_weights[true_sign < 0] = self.downward_weight
        
        # Calculate weighted sign match
        matches = (pred_sign == true_sign).float()
        weighted_matches = matches * dir_weights
        direction_loss = 1.0 - (weighted_matches.sum() / dir_weights.sum())
        
        # Combine losses
        combined_loss = (1 - self.alpha) * mse_loss + self.alpha * direction_loss
        
        return combined_loss

class DownwardSpecialistTFT(DirectionalTFT):
    """Extended TFT specialized for downward movement prediction"""
    
    def __init__(self, *args, alpha=0.7, downward_weight=3.0, **kwargs):
        super().__init__(*args, alpha=alpha, **kwargs)
        self.downward_weight = downward_weight
        self.downward_loss = DownwardFocusLoss(alpha=alpha, downward_weight=downward_weight)
        
        # Track downward prediction accuracy specifically
        self.down_acc_train = []
        self.down_acc_val = []
    
    def _apply_loss(self, y_pred, y_true):
        if self.loss_fn == "downward":
            return self.downward_loss(y_pred, y_true)
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
        if self.loss_fn in ["mse", "directional", "downward"]:
            predictions = predictions.squeeze(-1)
        
        # Calculate loss
        loss = self._apply_loss(predictions, targets)
        
        # Calculate direction accuracy (overall)
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(targets)
        direction_acc = (pred_sign == true_sign).float().mean()
        
        # Calculate downward direction accuracy specifically
        down_mask = (true_sign < 0)
        if down_mask.sum() > 0:
            down_acc = (pred_sign[down_mask] == true_sign[down_mask]).float().mean()
            self.log('train_down_acc', down_acc, prog_bar=True)
            self.down_acc_train.append(down_acc.item())
        
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
        if self.loss_fn in ["mse", "directional", "downward"]:
            predictions = predictions.squeeze(-1)
        
        loss = self._apply_loss(predictions, targets)
        
        # Direction accuracy (overall)
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(targets)
        direction_acc = (pred_sign == true_sign).float().mean()
        
        # Downward direction accuracy specifically
        down_mask = (true_sign < 0)
        if down_mask.sum() > 0:
            down_acc = (pred_sign[down_mask] == true_sign[down_mask]).float().mean()
            self.log('val_down_acc', down_acc, prog_bar=True)
            self.down_acc_val.append(down_acc.item())
        
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

def create_downward_focused_dataset(df, down_ratio=0.6, random_state=42):
    """Create a dataset with higher proportion of downward movements
    
    Args:
        df: Original dataframe
        down_ratio: Target ratio of downward examples (0-1)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with adjusted class distribution
    """
    # Get the sign of the target
    target_col = 'target_ma_adjusted' if 'target_ma_adjusted' in df.columns else 'target_sign'
    signs = np.sign(df[target_col])
    
    # Split by direction
    df_down = df[signs < 0]
    df_up = df[signs > 0]
    df_zero = df[signs == 0]
    
    logger.info(f"Original distribution: {len(df_down)} down, {len(df_up)} up, {len(df_zero)} zero")
    
    # Calculate desired sizes
    down_size = len(df_down)
    desired_total = int(down_size / down_ratio)
    desired_up_and_zero = desired_total - down_size
    
    # Calculate ratio of up vs zero in the original data
    if len(df_zero) > 0:
        up_zero_ratio = len(df_up) / (len(df_up) + len(df_zero))
        desired_up = int(desired_up_and_zero * up_zero_ratio)
        desired_zero = desired_up_and_zero - desired_up
    else:
        desired_up = desired_up_and_zero
        desired_zero = 0
    
    # Sample up and zero classes
    sampled_up = df_up.sample(n=min(len(df_up), desired_up), random_state=random_state)
    if len(df_zero) > 0 and desired_zero > 0:
        sampled_zero = df_zero.sample(n=min(len(df_zero), desired_zero), random_state=random_state)
    else:
        sampled_zero = pd.DataFrame()
    
    # Combine dataframes
    focused_df = pd.concat([df_down, sampled_up, sampled_zero])
    
    # Shuffle
    focused_df = focused_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate new distribution
    new_signs = np.sign(focused_df[target_col])
    logger.info(f"New distribution: {sum(new_signs < 0)} down ({sum(new_signs < 0)/len(focused_df):.2%}), "
                f"{sum(new_signs > 0)} up ({sum(new_signs > 0)/len(focused_df):.2%}), "
                f"{sum(new_signs == 0)} zero ({sum(new_signs == 0)/len(focused_df):.2%})")
    
    return focused_df

def train_downward_specialist(
    df,
    target_col='target_ma_adjusted',
    hidden_size=128,
    num_attention_heads=4,
    dropout=0.2,
    alpha=0.7,
    downward_weight=3.0,
    learning_rate=5e-4,
    batch_size=16,
    max_epochs=50,
    patience=5,
    context_length=20,
    use_gpu=True
):
    """Train the downward specialist TFT model"""
    # Prepare data
    train_dataset, val_dataset, test_dataset, feature_cols, scalers = prepare_data(
        df, 
        target_col=target_col,
        context_length=context_length,
        prediction_length=1
    )
    
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
    model = DownwardSpecialistTFT(
        static_variables=[],  # No static variables
        time_varying_categorical_variables=[],  # No categorical variables for now
        time_varying_real_variables=len(feature_cols),
        static_embedding_sizes=[],  # No static embeddings
        time_varying_embedding_sizes=[],  # No categorical embeddings
        hidden_size=hidden_size,
        lstm_layers=2,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        context_length=context_length,
        prediction_length=1,
        loss_fn="downward",  # Use our specialized downward loss
        alpha=alpha,
        downward_weight=downward_weight,
        bias_correction=True
    )
    
    # Create checkpoint callback
    checkpoint_dir = f"checkpoints/downward_tft_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tft-{epoch:02d}-{val_down_acc:.4f}',  # Monitor downward accuracy
        save_top_k=3,
        monitor='val_down_acc',  # We specifically want to maximize downward accuracy
        mode='max'
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_down_acc',
        patience=patience,
        mode='max'
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up logger
    logger_dir = f"lightning_logs/downward_tft_{timestamp}"
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
    logger.info(f"Starting training downward specialist with {len(train_loader.dataset)} samples")
    logger.info(f"Alpha: {alpha}, Downward weight: {downward_weight}")
    
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
    model_path = f"models/downward_specialist_tft_{timestamp}.pt"
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'downward_specialist_tft',
        'hyperparameters': {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'dropout': dropout,
            'alpha': alpha,
            'downward_weight': downward_weight,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'context_length': context_length
        },
        'num_features': len(feature_cols),
        'test_results': test_results[0] if test_results else None,
        'best_val_acc': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        'best_epoch': checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0] if checkpoint_callback.best_model_path else None
    }
    
    # Load best model if available
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        best_model = DownwardSpecialistTFT.load_from_checkpoint(best_model_path)
        
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

def plot_training_curves(model, metadata, plot_dir="plots"):
    """Plot training curves with focus on downward accuracy"""
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Overall direction accuracy
    plt.subplot(1, 2, 1)
    plt.plot(model.train_acc, label='Train')
    plt.plot(model.val_acc, label='Validation')
    plt.title('Overall Direction Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Downward direction accuracy
    plt.subplot(1, 2, 2)
    plt.plot(model.down_acc_train, label='Train')
    plt.plot(model.down_acc_val, label='Validation')
    plt.title('Downward Direction Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    timestamp = metadata['timestamp']
    plt.savefig(f"{plot_dir}/downward_specialist_curves_{timestamp}.png")
    logger.info(f"Training curves saved to {plot_dir}/downward_specialist_curves_{timestamp}.png")

def main():
    parser = argparse.ArgumentParser(description='Train TFT model specialized for downward movements')
    parser.add_argument('--target', type=str, default='target_ma_adjusted', 
                        choices=['target_sign', 'target_binned', 'target_log', 'target_ma_adjusted', 'original_target'],
                        help='Target column to use for training')
    parser.add_argument('--alpha', type=float, default=0.7, 
                        help='Balance between MSE (0) and directional accuracy (1)')
    parser.add_argument('--downward_weight', type=float, default=3.0, 
                        help='Weight for downward examples in the loss function')
    parser.add_argument('--down_ratio', type=float, default=0.6,
                        help='Ratio of downward examples in the focused dataset')
    parser.add_argument('--context', type=int, default=20, help='Context length (time steps)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training even if GPU is available')
    parser.add_argument('--dataset', type=str, default='data_improvements/improved_dataset_small.csv',
                       help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.dataset)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Create downward focused dataset
    logger.info("Creating downward-focused dataset...")
    focused_df = create_downward_focused_dataset(df, down_ratio=args.down_ratio)
    
    # Train model
    model, metadata = train_downward_specialist(
        df=focused_df,
        target_col=args.target,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        dropout=args.dropout,
        alpha=args.alpha,
        downward_weight=args.downward_weight,
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
    plot_training_curves(model, metadata)

if __name__ == "__main__":
    main() 