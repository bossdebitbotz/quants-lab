# TFT Model Training Protocol

## Overview

This document outlines the standard protocol for training Temporal Fusion Transformer (TFT) models for cryptocurrency price prediction. Following this protocol ensures reproducible results and proper model evaluation.

## Data Preparation

1. **Data Acquisition**:
   - Query 10-second bar data from PostgreSQL database
   - Minimum data requirement: 100,000 samples (~2 weeks of market data)
   - Ensure data covers various market regimes (trending, ranging, volatile)

2. **Feature Engineering**:
   - Calculate all features as defined in `features.md`
   - Verify no lookahead bias in feature calculations
   - Confirm data timestamps are properly aligned

3. **Data Splitting**:
   - **Training**: 70% of data (chronologically first)
   - **Validation**: 15% of data (immediately after training)
   - **Testing**: 15% of data (most recent)
   - Record exact timestamp ranges for reproducibility

4. **Preprocessing**:
   - Calculate normalization statistics (mean, std) on training data only
   - Apply normalization to all three sets using training statistics
   - Handle missing values consistently (fill with 0 after normalization)
   - Create DataLoader with appropriate batch size and context length

## Model Configuration

1. **Standard Architecture**:
   - Input features: All 21 features from `features.md`
   - Context length: 30 time steps (5 minutes of data)
   - Hidden dimension: 128
   - Attention heads: 4
   - Dropout rate: 0.2
   - LSTM layers: 2

2. **Hyperparameters**:
   - Learning rate: 1e-4
   - Batch size: 64
   - Epochs: Maximum 100 with early stopping
   - Loss function: MSE
   - Optimizer: Adam
   - Weight decay: 1e-6

3. **Configuration Storage**:
   - Save complete configuration as JSON file
   - Include all hyperparameters, feature list, and preprocessing steps
   - Store in `configs/` directory with timestamp

## Training Procedure

1. **Initialization**:
   - Set random seed (default: 42) for reproducibility
   - Initialize model with configuration
   - Move model to GPU if available
   - Set up TensorBoard or other logging

2. **Training Loop**:
   ```python
   # Setup
   model = TemporalFusionTransformer(**config)
   optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
   criterion = nn.MSELoss()
   
   # Training loop
   for epoch in range(config['max_epochs']):
       model.train()
       epoch_loss = 0
       
       for batch_x, batch_y in train_dataloader:
           # Move to device
           batch_x = batch_x.to(device)
           batch_y = batch_y.to(device)
           
           # Forward pass
           optimizer.zero_grad()
           outputs = model(None, batch_x)  # None for static features
           loss = criterion(outputs, batch_y)
           
           # Backward pass
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
           optimizer.step()
           
           epoch_loss += loss.item()
       
       # Validation
       model.eval()
       val_loss = 0
       
       with torch.no_grad():
           for val_x, val_y in val_dataloader:
               val_x = val_x.to(device)
               val_y = val_y.to(device)
               val_outputs = model(None, val_x)
               val_loss += criterion(val_outputs, val_y).item()
       
       # Logging
       print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_dataloader):.6f}, Val Loss = {val_loss/len(val_dataloader):.6f}")
       
       # Early stopping check
       # Save checkpoint if best
   ```

3. **Early Stopping**:
   - Monitor validation loss
   - Patience: 10 epochs
   - Save best model based on validation loss
   - Record training and validation curves

4. **Checkpointing**:
   - Save model checkpoints every 10 epochs
   - Save best model based on validation metrics
   - Include optimizer state and training progress
   - Record timestamp and performance metrics

## Bias Correction

1. **Bias Detection**:
   - Evaluate model on validation set
   - Calculate mean prediction error: `mean(y_pred - y_true)`
   - If absolute bias > 0.0001, apply correction

2. **Correction Procedure**:
   - Subtract bias constant from all predictions
   - Save bias-corrected model with suffix `_bias_corrected`
   - Document bias value in model metadata

3. **Verification**:
   - Re-evaluate corrected model on validation set
   - Confirm bias is now close to zero
   - Verify other metrics (RMSE, correlation) are maintained or improved

## Model Validation

1. **Performance Metrics**:
   - Calculate all metrics defined in `evaluation_metrics.md`
   - Compare against baseline models and previous versions
   - Record all metrics with model metadata

2. **Ablation Studies**:
   - Test variations of feature sets (optional)
   - Evaluate different hyperparameter settings
   - Document impact of architectural changes

3. **Trading Simulation**:
   - Run trading simulation with standard parameters
   - Record PnL, win rate, and other trading metrics
   - Compare to baseline trading strategies

## Model Persistence

1. **Saving Format**:
   ```python
   # Save complete model data
   torch.save({
       'model': model.state_dict(),
       'config': config,
       'bias_correction': bias_value,
       'performance': {
           'train_loss': final_train_loss,
           'val_loss': final_val_loss,
           'test_metrics': test_metrics
       },
       'feature_stats': {
           'means': feature_means,
           'stds': feature_stds
       }
   }, f"models/tft_{frequency}_{variant}_{timestamp}.pt")
   ```

2. **Documentation**:
   - Update `docs/changelog.md` with new model version
   - Document key findings and improvements
   - Note any issues or limitations discovered

3. **Versioning**:
   - Use semantic versioning for model releases
   - Major version: Significant architecture changes
   - Minor version: Feature or hyperparameter improvements
   - Patch version: Bug fixes or minor adjustments

## Reproducibility Requirements

1. **Environment**:
   - Record Python version and all package versions
   - Document hardware specifications (CPU, GPU, RAM)
   - Save environment configuration (requirements.txt or environment.yml)

2. **Code Versioning**:
   - Commit all code changes to version control
   - Tag releases with model version
   - Document any manual steps not captured in code

3. **Random Seeds**:
   - Set and document all random seeds (Python, NumPy, PyTorch)
   - Test with multiple seeds for robust evaluation
   - Report variability across different seeds 