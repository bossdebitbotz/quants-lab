import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemporalFusionTransformer(pl.LightningModule):
    def __init__(
        self,
        static_features: int = 0,
        time_varying_features: int = 8,  # Number of features in our dataset
        num_heads: int = 4,
        hidden_size: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        context_length: int = 10,
        prediction_length: int = 1,
        bias_correction: bool = False
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Architecture parameters
        self.static_features = static_features
        self.time_varying_features = time_varying_features
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.bias_correction = bias_correction
        self.prediction_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=bias_correction)
        
        # Variable selection networks
        self.static_var_selection = nn.Sequential(
            nn.Linear(static_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, static_features),
            nn.Softmax(dim=-1)
        ) if static_features > 0 else None
        
        # Feature transformation
        self.feature_layer = nn.Linear(time_varying_features, hidden_size)
        
        # Processing layers
        self.temporal_processing = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention layers
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Positional encoding
        self.positional_encoding = torch.nn.Parameter(
            torch.zeros(1, context_length, hidden_size)
        )
        
        # Transformer encoder layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )
        
        # Output layers
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_length)
        )
        
    def forward(
        self,
        static_features: torch.Tensor = None,
        temporal_features: torch.Tensor = None,
    ) -> torch.Tensor:
        # Ensure temporal_features has the right shape [batch_size, seq_len, n_features]
        if temporal_features.dim() != 3:
            raise ValueError(f"Expected temporal_features to have 3 dimensions, got {temporal_features.dim()}")
        
        batch_size, seq_len, _ = temporal_features.shape
        
        # Process temporal features
        temporal_embedded = self.feature_layer(temporal_features)
        
        # Add positional encoding
        temporal_embedded = temporal_embedded + self.positional_encoding[:, :seq_len, :]
        
        # Process static features if provided
        if static_features is not None and self.static_features > 0:
            static_embedded = self.static_var_selection(static_features)
            static_embedded = static_embedded.unsqueeze(1).expand(-1, seq_len, -1)
            combined_features = temporal_embedded + static_embedded
        else:
            combined_features = temporal_embedded
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(combined_features)
        
        # Get the final state for prediction (last time step)
        final_state = encoder_output[:, -1, :]
        
        # Generate predictions
        predictions = self.final_layer(final_state)
        
        # Apply bias correction if enabled
        if self.bias_correction:
            predictions = predictions - self.prediction_bias
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        static_features = batch.get('static_features')
        temporal_features = batch['temporal_features']
        targets = batch['targets']
        
        predictions = self(static_features, temporal_features)
        loss = nn.MSELoss()(predictions, targets)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        static_features = batch.get('static_features')
        temporal_features = batch['temporal_features']
        targets = batch['targets']
        
        predictions = self(static_features, temporal_features)
        loss = nn.MSELoss()(predictions, targets)
        
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        static_features = batch.get('static_features')
        temporal_features = batch['temporal_features']
        targets = batch['targets']
        
        predictions = self(static_features, temporal_features)
        loss = nn.MSELoss()(predictions, targets)
        
        # Log additional metrics for test set
        self.log('test_loss', loss)
        
        # Calculate and log prediction error
        error = torch.abs(predictions - targets).mean()
        self.log('test_mae', error)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

def collate_fn(batch):
    """Custom collate function to handle batching of temporal sequences."""
    # Stack all temporal features
    temporal_features = torch.stack([item['temporal_features'] for item in batch], dim=0)
    targets = torch.stack([item['targets'] for item in batch], dim=0)
    
    output = {
        'temporal_features': temporal_features,
        'targets': targets
    }
    
    # Handle static features if they exist
    if 'static_features' in batch[0]:
        static_features = torch.stack([item['static_features'] for item in batch], dim=0)
        output['static_features'] = static_features
    
    return output

class MarketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int = 10,
        prediction_length: int = 1,
        static_features: List[str] = None,
        target_column: str = 'target'
    ):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.static_features = static_features or []
        self.target_column = target_column
        
        # Validate data
        if len(data) < context_length + prediction_length:
            raise ValueError(f"Dataset length ({len(data)}) must be greater than "
                          f"context_length + prediction_length ({context_length + prediction_length})")
        
        # Ensure all required columns exist
        required_columns = set(self.static_features + [self.target_column])
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate number of valid samples
        self.num_samples = max(0, len(data) - context_length - prediction_length + 1)
        logger.info(f"Created dataset with {self.num_samples} samples")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.num_samples} samples")
        
        # Get the window of data
        window_start = idx
        window_end = idx + self.context_length + self.prediction_length
        window_data = self.data.iloc[window_start:window_end].copy()
        
        # Split into context and target periods
        context_data = window_data.iloc[:self.context_length]
        target_data = window_data.iloc[-self.prediction_length:][self.target_column]
        
        # Prepare features
        feature_cols = [col for col in context_data.columns if col not in self.static_features + [self.target_column]]
        temporal_features = context_data[feature_cols].astype(float)
        static_features = context_data[self.static_features].iloc[0].astype(float) if self.static_features else None
        
        # Convert to tensors and ensure correct shapes
        temporal_tensor = torch.FloatTensor(temporal_features.values)  # Shape: [context_length, n_features]
        target_tensor = torch.FloatTensor(target_data.values).unsqueeze(-1)  # Shape: [prediction_length, 1]
        
        sample = {
            'temporal_features': temporal_tensor,
            'targets': target_tensor
        }
        
        if static_features is not None:
            static_tensor = torch.FloatTensor(static_features.values)
            sample['static_features'] = static_tensor
        
        return sample

def create_data_loaders(
    data: pd.DataFrame,
    context_length: int = 10,
    prediction_length: int = 1,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    static_features: List[str] = None,
    target_column: str = 'target'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Validate inputs
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    if len(data) < context_length + prediction_length:
        raise ValueError(f"Not enough data points. Need at least {context_length + prediction_length}, got {len(data)}")
    
    # Calculate minimum required samples for each split
    min_samples = context_length + prediction_length
    
    # Adjust split ratios if necessary
    total_samples = len(data)
    if total_samples < 3 * min_samples:
        logger.warning(f"Small dataset detected ({total_samples} samples). Adjusting split ratios.")
        # Ensure each split has at least min_samples
        train_size = min(total_samples - 2 * min_samples, int(total_samples * 0.6))
        val_size = min(min_samples, int(total_samples * 0.2))
    else:
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
    
    # Create splits
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    
    logger.info(f"Data split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Validate split sizes
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        if len(split_data) < context_length + prediction_length:
            logger.warning(f"{split_name} split has insufficient samples ({len(split_data)}). Using full dataset for all splits.")
            # Use the full dataset for all splits when individual splits are too small
            train_data = val_data = test_data = data
            break
    
    # Create datasets
    train_dataset = MarketDataset(
        train_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_features=static_features,
        target_column=target_column
    )
    
    val_dataset = MarketDataset(
        val_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_features=static_features,
        target_column=target_column
    )
    
    test_dataset = MarketDataset(
        test_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_features=static_features,
        target_column=target_column
    )
    
    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader 