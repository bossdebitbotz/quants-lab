import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network as described in the TFT paper.
    This allows the model to optionally skip connections based on input data.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
        residual: bool = True
    ):
        """
        Gated Residual Network as described in the TFT paper
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (if None, uses input_size)
            dropout: Dropout rate
            context_size: Optional context dimension for conditioning
            residual: Whether to use residual connection (if dimensions match)
        """
        super().__init__()
        
        # If output_size is not specified, default to input_size for residual connection
        self.output_size = output_size or input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        # Flag for using residual connection (only possible if input_size matches output_size)
        self.residual = residual and (input_size == self.output_size)
        
        # Main dense layer mapping input to hidden dimension
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Optional context projection if context is to be included
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        
        # Second dense layer mapping hidden to output dimension
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        
        # Gating layer for selecting portions of the main branch vs. residual
        self.gate = nn.Linear(input_size, self.output_size)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.elu = nn.ELU()
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Gated Residual Network
        
        Args:
            x: Input tensor of shape (..., input_size)
            context: Optional context tensor of shape (..., context_size)
            
        Returns:
            Tensor of shape (..., output_size)
        """
        # Store original input for residual connection if needed
        residual = x
        
        # First linear layer
        hidden = self.fc1(x)
        
        # Add context if provided
        if self.context_size is not None and context is not None:
            # Make sure context has same batch dims as x
            if context.shape[:-1] != x.shape[:-1]:
                # Expand context to match batch dims of x, preserving context features
                expanded_shape = list(x.shape[:-1]) + [context.shape[-1]]
                context = context.expand(expanded_shape)
            
            # Project context and add to hidden representation
            context_projection = self.context_proj(context)
            hidden = hidden + context_projection
        
        # Apply ELU activation and dropout
        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        
        # Second linear layer
        hidden = self.fc2(hidden)
        
        # Apply gating mechanism
        gate = torch.sigmoid(self.gate(x))
        
        # Combine gated output with residual if possible
        if self.residual:
            output = gate * hidden + (1 - gate) * residual
        else:
            # If residual connection is not possible (input_size != output_size)
            # just use gated output
            output = gate * hidden
        
        # Apply layer normalization
        output = self.norm(output)
        
        return output

class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: List[int],
        hidden_size: int,
        dropout: float = 0.1,
        output_size: Optional[int] = None,
    ):
        """
        Variable Selection Network as described in the TFT paper
        
        Args:
            input_sizes: List of input sizes for each variable
            hidden_size: Hidden layer size for the network
            dropout: Dropout rate
            output_size: Output size (if None, uses hidden_size)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_variables = len(input_sizes)
        
        if output_size is None:
            output_size = hidden_size
        
        # Ensure all inputs are transformed to hidden_size for variable selection
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            ) for input_size in input_sizes
        ])
        
        # Create GRN for variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=self.num_variables,
            dropout=dropout
        )
        
        # Final processing of combined outputs
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )
        
    def forward(
        self,
        variables: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Variable Selection Network
        
        Args:
            variables: List of input variables each with shape (..., input_size_i)
            context: Optional context with shape (..., hidden_size)
            
        Returns:
            Tuple of (processed_output, selection_weights)
        """
        # Check that we have the right number of variables
        if len(variables) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {len(variables)}")
        
        # Get the common batch dimensions from first variable
        batch_dims = variables[0].shape[:-1]
        
        # Process each variable with its own GRN
        processed_variables = []
        for i, (variable, grn) in enumerate(zip(variables, self.variable_grns)):
            # Ensure the variable has the expected input size
            if variable.size(-1) != self.input_sizes[i]:
                raise ValueError(
                    f"Variable {i} expected input size {self.input_sizes[i]}, "
                    f"got {variable.size(-1)}"
                )
            
            # Process through GRN
            processed = grn(variable)
            processed_variables.append(processed)
        
        # Stack processed variables
        stacked = torch.stack(processed_variables, dim=-2)  # Shape: (..., num_vars, hidden_size)
        
        # Get selection weights based on context (if provided)
        if context is not None:
            # Add context for weight calculation
            selection_weights = self.selection_grn(context)
        else:
            # Use only the variable information for weight calculation
            # Use the first variable's shape to determine batch dimensions
            flat_batch_size = np.prod(batch_dims)
            zeros = torch.zeros((flat_batch_size, self.hidden_size), 
                               device=variables[0].device)
            selection_weights = self.selection_grn(zeros)
        
        # Apply softmax to get weights summing to 1
        selection_weights = torch.softmax(selection_weights, dim=-1)
        
        # Reshape for broadcasting with stacked variables if needed
        if selection_weights.dim() < stacked.dim():
            # Add variable dimension to selection weights
            for _ in range(stacked.dim() - selection_weights.dim() - 1):
                selection_weights = selection_weights.unsqueeze(1)
            # Add hidden_size dimension
            selection_weights = selection_weights.unsqueeze(-1)
        else:
            # Just add the hidden_size dimension
            selection_weights = selection_weights.unsqueeze(-1)
        
        # Apply weights to variables
        selected = (stacked * selection_weights).sum(dim=-2)  # Sum over variables dim
        
        # Final processing through output GRN
        output = self.output_grn(selected)
        
        # Remove the last dimension from selection_weights for return
        selection_weights = selection_weights.squeeze(-1)
        
        return output, selection_weights

class TemporalFusionTransformer(pl.LightningModule):
    def __init__(
        self,
        static_variables: List[int] = None,  # List of dimensions for each static variable
        time_varying_categorical_variables: List[int] = None,  # List of dimensions for each categorical variable
        time_varying_real_variables: int = 1,  # Number of continuous variables
        static_embedding_sizes: List[Tuple[int, int]] = None,  # Embedding sizes for static categorical variables
        time_varying_embedding_sizes: List[Tuple[int, int]] = None,  # Embedding sizes for time-varying categorical variables
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        context_length: int = 30,  # Length of historical context
        prediction_length: int = 1,  # Number of future steps to predict
        quantiles: List[float] = [0.1, 0.5, 0.9],  # Quantiles for probabilistic forecasting
        loss_fn: str = "quantile",  # Options: "quantile" or "mse"
        bias_correction: bool = True
    ):
        super().__init__()
        
        # Save hyperparameters for easy loading
        self.save_hyperparameters()
        
        # Setup parameters
        self.static_variables = static_variables or []
        self.time_varying_categorical_variables = time_varying_categorical_variables or []
        self.time_varying_real_variables = time_varying_real_variables
        self.static_embedding_sizes = static_embedding_sizes or []
        self.time_varying_embedding_sizes = time_varying_embedding_sizes or []
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.loss_fn = loss_fn
        self.bias_correction = bias_correction
        
        # Setup variable dimensions
        self.static_embedding_dim = sum([size[1] for size in self.static_embedding_sizes]) if self.static_embedding_sizes else 0
        self.temporal_embedding_dim = sum([size[1] for size in self.time_varying_embedding_sizes]) if self.time_varying_embedding_sizes else 0
        self.temporal_variable_count = len(self.time_varying_categorical_variables) + 1  # +1 for real variables
        
        # Create embeddings for categorical variables
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(classes, dim) 
            for classes, dim in self.static_embedding_sizes
        ]) if self.static_embedding_sizes else None
        
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(classes, dim) 
            for classes, dim in self.time_varying_embedding_sizes
        ]) if self.time_varying_embedding_sizes else None
        
        # Create variable selection networks
        if self.static_embedding_dim > 0:
            self.static_var_selection = VariableSelectionNetwork(
                input_sizes=[dim for _, dim in self.static_embedding_sizes],
                hidden_size=self.hidden_size,
                dropout=self.dropout
            )
        
        self.temporal_var_selection = VariableSelectionNetwork(
            input_sizes=[self.hidden_size] * self.temporal_variable_count,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            output_size=self.hidden_size if self.static_embedding_dim > 0 else None
        )
        
        # Input transformation layers
        self.static_context_grn = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        ) if self.static_embedding_dim > 0 else None
        
        self.real_transform = nn.Linear(self.time_varying_real_variables, self.hidden_size)
        
        # Temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention components
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=self.num_attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Feature processing after attention
        self.post_attention_grn = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )
        
        # Position-wise feed-forward
        self.pos_wise_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )
        
        # Final output layer
        output_size = len(self.quantiles) if self.loss_fn == "quantile" else 1
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        
        # Bias correction
        if self.bias_correction:
            self.prediction_bias = nn.Parameter(torch.zeros(output_size), requires_grad=True)
    
    def forward(
        self,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        temporal_categorical_features: Optional[torch.Tensor] = None,
        temporal_real_features: torch.Tensor = None,
    ):
        """
        Forward pass through the Temporal Fusion Transformer
        
        Args:
            static_categorical_features: Optional tensor of shape (batch_size, num_static_categorical)
            static_real_features: Optional tensor of shape (batch_size, num_static_real)
            temporal_categorical_features: Optional tensor of shape (batch_size, seq_len, num_temporal_categorical)
            temporal_real_features: Tensor of shape (batch_size, seq_len, num_temporal_real)
            
        Returns:
            Dictionary containing:
                - prediction: Tensor of shape (batch_size, prediction_length, output_size)
                - attention_weights: Attention weights
                - temporal_weights: Temporal variable selection weights
                - static_weights: Static variable selection weights (if applicable)
        """
        if temporal_real_features is None:
            raise ValueError("temporal_real_features cannot be None")
            
        # Get batch and sequence dimensions
        batch_size = temporal_real_features.size(0)
        seq_len = temporal_real_features.size(1)
        
        # Process static inputs (if any)
        static_context = None
        static_weights = None
        if self.static_embedding_dim > 0 and static_categorical_features is not None:
            # Process each static categorical variable through its embedding
            static_embedded = []
            for i, embedding in enumerate(self.static_embeddings):
                # Get the i-th static categorical feature and apply embedding
                static_cat = static_categorical_features[..., i]
                embedded = embedding(static_cat)
                static_embedded.append(embedded)
                
            # Apply static variable selection
            static_embedding, static_weights = self.static_var_selection(static_embedded)
            
            # Generate static context vector
            static_context = self.static_context_grn(static_embedding)
        
        # Process temporal categorical inputs (if any)
        temporal_embedded = []
        if temporal_categorical_features is not None and self.temporal_embeddings:
            # Process each temporal categorical variable
            for i, embedding in enumerate(self.temporal_embeddings):
                # Get the i-th temporal categorical feature across sequence
                temporal_cat = temporal_categorical_features[..., i]
                
                # Apply embedding and reshape to (batch_size, seq_len, embedding_dim)
                embedded = embedding(temporal_cat)
                if embedded.dim() < 3:
                    embedded = embedded.view(batch_size, seq_len, -1)
                    
                temporal_embedded.append(embedded)
        
        # Process temporal real inputs
        # Transform to same dimension as categorical embeddings for consistency
        if temporal_real_features.size(2) != self.hidden_size:
            real_embedded = self.real_transform(temporal_real_features)
        else:
            real_embedded = temporal_real_features
            
        temporal_embedded.append(real_embedded)
        
        # Prepare inputs for variable selection - reshape for flat processing
        temporal_input = []
        for embed in temporal_embedded:
            # Reshape to (batch_size * seq_len, hidden_size) for variable selection
            flat_embed = embed.view(batch_size * seq_len, -1)
            temporal_input.append(flat_embed)
        
        # Prepare static context for temporal processing if available
        expanded_static_context = None
        if static_context is not None:
            # Expand static context for each time step
            expanded_static_context = static_context.unsqueeze(1).repeat(1, seq_len, 1)
            # Reshape to (batch_size * seq_len, hidden_size) to match temporal inputs
            expanded_static_context = expanded_static_context.view(batch_size * seq_len, -1)
            
        # Apply temporal variable selection with optional static context
        temporal_features, temporal_weights = self.temporal_var_selection(
            temporal_input, expanded_static_context
        )
        
        # Reshape back to sequence form
        temporal_features = temporal_features.view(batch_size, seq_len, self.hidden_size)
        
        # Split into encoder and decoder parts
        if seq_len < self.context_length + self.prediction_length:
            # For evaluation, all steps might be used for context
            encoder_steps = min(self.context_length, seq_len)
            decoder_steps = seq_len - encoder_steps
        else:
            encoder_steps = self.context_length
            decoder_steps = self.prediction_length
            
        encoder_input = temporal_features[:, :encoder_steps, :]
        
        if decoder_steps > 0:
            decoder_input = temporal_features[:, encoder_steps:encoder_steps + decoder_steps, :]
        else:
            # Handle case where no decoder steps are available
            decoder_input = torch.zeros((batch_size, 1, self.hidden_size), device=temporal_features.device)
        
        # Encoder LSTM
        encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_input)
        
        # Decoder LSTM
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))
        
        # Attention mechanism
        attn_input = torch.cat([encoder_output, decoder_output], dim=1)
        attn_output, attn_weights = self.attention(
            query=decoder_output,
            key=attn_input,
            value=attn_input
        )
        
        # Skip connection over attention with gated residual network
        processed_attn = self.post_attention_grn(attn_output)
        
        # Position-wise feed-forward with gated residual network
        output = self.pos_wise_ff(processed_attn)
        
        # Final output layer
        final_output = self.output_layer(output)
        
        # Apply bias correction if enabled
        if self.bias_correction:
            final_output = final_output - self.prediction_bias
        
        # Return predictions and attention information
        result = {
            'prediction': final_output,
            'attention_weights': attn_weights,
            'temporal_weights': temporal_weights,
            'static_weights': static_weights
        }
        
        return result
    
    def _apply_loss(self, y_pred, y_true):
        if self.loss_fn == "mse":
            return nn.MSELoss()(y_pred, y_true)
        elif self.loss_fn == "quantile":
            # Quantile loss implementation
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = y_true - y_pred[..., i]
                loss = torch.max((q - 1) * errors, q * errors).mean()
                losses.append(loss)
            return sum(losses) / len(losses)
    
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
        if self.loss_fn == "mse":
            predictions = predictions.squeeze(-1)
        
        # Calculate loss
        loss = self._apply_loss(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
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
        if self.loss_fn == "mse":
            predictions = predictions.squeeze(-1)
        
        # Calculate loss
        loss = self._apply_loss(predictions, targets)
        
        # Calculate additional metrics
        if self.loss_fn == "mse":
            mae = torch.abs(predictions - targets).mean()
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            self.log('val_mae', mae, prog_bar=True)
            self.log('val_rmse', rmse, prog_bar=True)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Same as validation step
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
        if self.loss_fn == "mse":
            predictions = predictions.squeeze(-1)
        
        # Calculate loss
        loss = self._apply_loss(predictions, targets)
        
        # Calculate additional metrics (more comprehensive for test)
        if self.loss_fn == "mse":
            mae = torch.abs(predictions - targets).mean()
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
            bias = (predictions - targets).mean()
            
            # Direction accuracy
            pred_direction = torch.sign(predictions)
            true_direction = torch.sign(targets)
            direction_accuracy = (pred_direction == true_direction).float().mean()
            
            self.log('test_mae', mae)
            self.log('test_rmse', rmse)
            self.log('test_bias', bias)
            self.log('test_direction_accuracy', direction_accuracy)
        
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        # Adam optimizer with learning rate scheduling
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class ImprovedMarketDataset(torch.utils.data.Dataset):
    """Enhanced dataset class that properly handles static and temporal features"""
    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int = 30,
        prediction_length: int = 1,
        static_categorical_cols: List[str] = None,
        static_real_cols: List[str] = None,
        temporal_categorical_cols: List[str] = None,
        temporal_real_cols: List[str] = None,
        target_col: str = 'target'
    ):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.static_categorical_cols = static_categorical_cols or []
        self.static_real_cols = static_real_cols or []
        self.temporal_categorical_cols = temporal_categorical_cols or []
        self.temporal_real_cols = temporal_real_cols or []
        self.target_col = target_col
        
        # Validate data
        if len(data) < context_length + prediction_length:
            raise ValueError(f"Dataset length ({len(data)}) must be greater than "
                             f"context_length + prediction_length ({context_length + prediction_length})")
        
        # Ensure all required columns exist
        required_columns = set(self.static_categorical_cols + self.static_real_cols + 
                               self.temporal_categorical_cols + self.temporal_real_cols + 
                               [self.target_col])
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate valid samples
        self.num_samples = max(0, len(data) - context_length - prediction_length + 1)
        logger.info(f"Created dataset with {self.num_samples} samples")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")
        
        # Get windows for historical and future data
        history_window = slice(idx, idx + self.context_length)
        future_window = slice(idx + self.context_length, idx + self.context_length + self.prediction_length)
        
        # Extract historical data
        history_data = self.data.iloc[history_window]
        
        # Extract future data
        future_data = self.data.iloc[future_window]
        
        # Create sample dict
        sample = {}
        
        # Process static features (use only first row since they're static)
        if self.static_categorical_cols:
            sample['static_categorical_features'] = torch.tensor(
                history_data[self.static_categorical_cols].iloc[0].values,
                dtype=torch.long
            )
        
        if self.static_real_cols:
            sample['static_real_features'] = torch.tensor(
                history_data[self.static_real_cols].iloc[0].values,
                dtype=torch.float32
            )
        
        # Process temporal features for history
        if self.temporal_categorical_cols:
            sample['temporal_categorical_features'] = torch.tensor(
                history_data[self.temporal_categorical_cols].values,
                dtype=torch.long
            )
        
        if self.temporal_real_cols:
            sample['temporal_real_features'] = torch.tensor(
                history_data[self.temporal_real_cols].values,
                dtype=torch.float32
            )
        
        # Process targets from the future window
        sample['targets'] = torch.tensor(
            future_data[self.target_col].values,
            dtype=torch.float32
        )
        
        return sample

def create_improved_data_loaders(
    data: pd.DataFrame,
    context_length: int = 30,
    prediction_length: int = 1,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    static_categorical_cols: List[str] = None,
    static_real_cols: List[str] = None,
    temporal_categorical_cols: List[str] = None,
    temporal_real_cols: List[str] = None,
    target_col: str = 'target',
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create DataLoaders for train, validation, and test datasets"""
    
    # Validate inputs
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    
    # Calculate split indices
    n_samples = len(data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    test_size = n_samples - train_size - val_size
    
    # Split data while preserving temporal order
    train_data = data.iloc[:train_size].copy()
    val_data = data.iloc[train_size:train_size + val_size].copy()
    test_data = data.iloc[train_size + val_size:].copy()
    
    # Log dataset sizes
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Create datasets
    train_dataset = ImprovedMarketDataset(
        data=train_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_categorical_cols=static_categorical_cols,
        static_real_cols=static_real_cols,
        temporal_categorical_cols=temporal_categorical_cols,
        temporal_real_cols=temporal_real_cols,
        target_col=target_col
    )
    
    val_dataset = ImprovedMarketDataset(
        data=val_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_categorical_cols=static_categorical_cols,
        static_real_cols=static_real_cols,
        temporal_categorical_cols=temporal_categorical_cols,
        temporal_real_cols=temporal_real_cols,
        target_col=target_col
    )
    
    test_dataset = ImprovedMarketDataset(
        data=test_data,
        context_length=context_length,
        prediction_length=prediction_length,
        static_categorical_cols=static_categorical_cols,
        static_real_cols=static_real_cols,
        temporal_categorical_cols=temporal_categorical_cols,
        temporal_real_cols=temporal_real_cols,
        target_col=target_col
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader 