# TFT Model Architecture

## Overview

The Temporal Fusion Transformer (TFT) is a sequence-to-sequence architecture specifically designed for multi-horizon time series forecasting. Our implementation is based on the original paper by Lim et al. (2019) with modifications tailored for high-frequency cryptocurrency price prediction.

## Key Components

### 1. Feature Processing Layer

- **Input Dimensions**: Variable number of time-varying features (currently 21)
- **Output Dimensions**: Mapped to hidden size (currently 128)
- **Implementation**: Linear layer followed by activation function

```python
self.feature_layer = nn.Linear(time_varying_features, hidden_size)
```

### 2. Temporal Processing

- **Architecture**: Bidirectional LSTM layers
- **Number of Layers**: 2
- **Hidden Dimensions**: 128
- **Implementation**: 

```python
self.temporal_processing = nn.LSTM(
    input_size=hidden_size,
    hidden_size=hidden_size,
    num_layers=2,
    batch_first=True,
    bidirectional=False  # Using unidirectional LSTM for causal modeling
)
```

### 3. Self-Attention Mechanism

- **Type**: Multi-head self-attention
- **Number of Heads**: 4
- **Implementation**:

```python
self.self_attention = nn.MultiheadAttention(
    embed_dim=hidden_size,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True
)
```

### 4. Final Prediction Layer

- **Architecture**: MLP with dropout
- **Output**: Single regression value (next period return)
- **Implementation**:

```python
self.final_layer = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, prediction_length)
)
```

### 5. Bias Correction (Post-Processing)

- **Description**: A constant value subtracted from raw predictions to correct systematic bias
- **Current Value**: 0.0002
- **Implementation**: Applied during inference

```python
def predict_with_bias_correction(x):
    with torch.no_grad():
        raw_prediction = model(None, x)
        return raw_prediction - bias_correction
```

## Model Parameters

- **Hidden Size**: 128
- **Number of Attention Heads**: 4
- **Dropout Rate**: 0.2
- **Learning Rate**: 1e-4
- **Context Length**: 30 (time steps used for prediction)
- **Prediction Length**: 1 (number of future time steps to predict)

## Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Batch Size**: 64
- **Normalization**: Feature-wise z-score normalization (mean=0, std=1)
- **Missing Value Handling**: Filled with zeros after normalization

## Custom Dataset Implementation

```python
class NumpyMarketDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, context_length=10):
        self.features = features
        self.targets = targets
        self.context_length = context_length
        self.n_samples = max(0, len(features) - context_length)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        feature_window = self.features[idx:idx+self.context_length]
        target = self.targets[idx+self.context_length-1]
        
        feature_tensor = torch.tensor(feature_window, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32).reshape(1)
        
        return feature_tensor, target_tensor
```

## Model Adaptations for Cryptocurrency Data

1. **No Static Features**: Unlike the original TFT paper, our implementation focuses exclusively on time-varying features
2. **Simplified Variable Selection**: We use a single linear projection rather than the gating mechanism in the original paper
3. **Bias Correction**: Added post-processing to address systematic prediction bias observed in cryptocurrency data
4. **Focused Prediction Horizon**: Single-step prediction (next 10-second bar) rather than multi-horizon

## Future Architecture Improvements

- Implement full variable selection network from the original paper
- Experiment with different temporal processing architectures (GRU, Transformer encoder)
- Add quantile regression capabilities for uncertainty estimation
- Incorporate static/categorical features for market regime conditioning
- Test performance impact of varying context lengths 