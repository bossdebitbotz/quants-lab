#!/usr/bin/env python3

"""
Custom TFT implementation for backtesting

This file contains a simplified version of the Temporal Fusion Transformer
model that is compatible with the saved model files in models/ directory.
"""

import torch
import torch.nn as nn
import numpy as np

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer implementation compatible with saved models.
    """
    def __init__(self, 
                 time_varying_real_variables=64,
                 static_variables=None,
                 time_varying_categorical_variables=None,
                 static_embedding_sizes=None,
                 time_varying_embedding_sizes=None,
                 hidden_size=128, 
                 lstm_layers=2,
                 num_attention_heads=4, 
                 dropout=0.1, 
                 context_length=30, 
                 prediction_length=1,
                 quantiles=None,
                 alpha=0.9,
                 bias_correction=False,
                 learning_rate=1e-4,
                 loss_fn="mse",
                 **kwargs):
        """
        Initialize the TFT model.
        
        Args:
            time_varying_real_variables: Number of continuous time-varying features
            static_variables: List of static variables (not used in this implementation)
            time_varying_categorical_variables: List of categorical variables (not used)
            static_embedding_sizes: Sizes for static embeddings (not used)
            time_varying_embedding_sizes: Sizes for time-varying embeddings (not used)
            hidden_size: Hidden dimension size
            lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
            context_length: Length of context window
            prediction_length: Number of future steps to predict
            quantiles: Quantiles for quantile regression (not used)
            alpha: Weight parameter for directional loss
            bias_correction: Whether to use bias correction
            learning_rate: Learning rate for training
            loss_fn: Loss function ("mse" or "directional")
        """
        super().__init__()
        
        # Store configuration
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_varying_real_variables = time_varying_real_variables
        self.bias_correction = bias_correction
        self.loss_fn = loss_fn
        self.alpha = alpha
        
        # For compatibility with saved models
        self.static_variables = static_variables or []
        self.time_varying_categorical_variables = time_varying_categorical_variables or []
        self.static_embedding_sizes = static_embedding_sizes or []
        self.time_varying_embedding_sizes = time_varying_embedding_sizes or []
        
        # Feature transformation layers
        self.feature_layer = nn.Linear(time_varying_real_variables, hidden_size)
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, prediction_length)
    
    def forward(self, 
                temporal_real_features=None,
                static_categorical_features=None,
                static_real_features=None,
                temporal_categorical_features=None,
                x=None,
                **kwargs):
        """
        Forward pass through the TFT model.
        
        Args:
            temporal_real_features: Time-varying continuous features [batch, seq_len, features]
            static_categorical_features: Static categorical features (not used)
            static_real_features: Static continuous features (not used)
            temporal_categorical_features: Time-varying categorical features (not used)
            x: Alternative input tensor (for compatibility)
            
        Returns:
            Dictionary with predictions
        """
        # Handle different input formats for compatibility
        if x is not None:
            # If x is directly provided, use it
            temporal_features = x
        elif temporal_real_features is not None:
            # If temporal_real_features is provided, use it
            temporal_features = temporal_real_features
        else:
            # If neither is provided, error
            raise ValueError("Either x or temporal_real_features must be provided")
        
        # Process features
        batch_size, seq_len, _ = temporal_features.shape
        
        # Feature transformation
        x = self.feature_layer(temporal_features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        # Transpose for attention: [seq_len, batch, hidden]
        lstm_out_t = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(
            query=lstm_out_t,
            key=lstm_out_t,
            value=lstm_out_t
        )
        # Transpose back: [batch, seq_len, hidden]
        attn_out = attn_out.transpose(0, 1)
        
        # Use the last timestep for prediction
        final_hidden = attn_out[:, -1]
        
        # Generate prediction
        x = self.fc1(final_hidden)
        x = torch.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        
        # Return predictions in the expected format
        if self.bias_correction:
            return {
                'prediction': x,
                'attention_weights': None  # Placeholder for attention weights
            }
        else:
            return {
                'prediction': x
            }
    
    def predict(self, temporal_features):
        """
        Make predictions using the model.
        
        Args:
            temporal_features: Input features [batch, seq_len, features]
            
        Returns:
            numpy array of predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x=temporal_features)
            if isinstance(output, dict):
                predictions = output['prediction']
            else:
                predictions = output
            return predictions.cpu().numpy()

class DirectionalTFT(TemporalFusionTransformer):
    """
    Extended TFT model with focus on directional accuracy.
    This version is compatible with the directional model saved in models/.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional attributes specific to directional model
        self.loss_fn = kwargs.get('loss_fn', 'directional')
    
    def forward(self, *args, **kwargs):
        """Forward pass through the directional TFT model"""
        return super().forward(*args, **kwargs) 