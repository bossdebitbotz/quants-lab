import os
import sys
import argparse
import logging
import traceback
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import List, Dict

# Import the model class and data loading functionality
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.improved_tft import TemporalFusionTransformer, ImprovedMarketDataset
from train_improved_tft import load_data, add_derived_features, prepare_data_for_tft

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_improved_evaluator")

class DatasetForEvaluation(torch.utils.data.Dataset):
    """Dataset class that provides both targets and input features for evaluation"""
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'returns_10sec',
        context_length: int = 30,
        prediction_length: int = 1
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Calculate valid samples
        self.num_samples = max(0, len(data) - context_length - prediction_length + 1)
        logger.info(f"Created evaluation dataset with {self.num_samples} samples")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get windows for historical and future data
        history_window = slice(idx, idx + self.context_length)
        future_window = slice(idx + self.context_length, idx + self.context_length + self.prediction_length)
        
        # Extract historical data
        history_data = self.data.iloc[history_window]
        
        # Extract future data
        future_data = self.data.iloc[future_window]
        
        # Create tensor for temporal_real_features
        temporal_real = torch.tensor(
            history_data[self.feature_cols].values,
            dtype=torch.float32
        )
        
        # Create tensor for targets
        targets = torch.tensor(
            future_data[self.target_col].values,
            dtype=torch.float32
        )
        
        return {
            'temporal_real_features': temporal_real,
            'targets': targets
        }

def load_model(model_path):
    """
    Load a saved TFT model from a file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (model, metadata, feature_definitions)
    """
    try:
        # Load with weights_only=False since we need the full model structure and metadata
        # Only use this with trusted model files
        model_package = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        model_state_dict = model_package['model_state_dict']
        model_config = model_package['config']
        metadata = model_package.get('metadata', {})
        feature_definitions = model_package.get('feature_definitions', {})
        
        # Extract configuration parameters
        hidden_size = model_config.get('hidden_size', 128)
        num_attention_heads = model_config.get('num_attention_heads', 4)
        dropout = model_config.get('dropout', 0.1)
        context_length = model_config.get('context_length', 30)
        prediction_length = model_config.get('prediction_length', 1)
        learning_rate = model_config.get('learning_rate', 1e-3)
        
        # Handle different model config formats
        static_variables = model_config.get('static_variables', [])
        time_varying_categorical_variables = model_config.get('time_varying_categorical_variables', [])
        time_varying_real_variables = model_config.get('time_varying_real_variables', 
                      len(feature_definitions.get('temporal_real_features', [])))
        
        static_embedding_sizes = model_config.get('static_embedding_sizes', [])
        time_varying_embedding_sizes = model_config.get('time_varying_embedding_sizes', [])
        
        # Create a new model instance with the saved configuration
        model = TemporalFusionTransformer(
            static_variables=static_variables,
            time_varying_categorical_variables=time_varying_categorical_variables,
            time_varying_real_variables=time_varying_real_variables,
            static_embedding_sizes=static_embedding_sizes,
            time_varying_embedding_sizes=time_varying_embedding_sizes,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            learning_rate=learning_rate,
            context_length=context_length,
            prediction_length=prediction_length,
            loss_fn="mse"  # Use MSE loss for evaluation
        )
        
        # Load the state dictionary into the model
        model.load_state_dict(model_state_dict)
        model.eval()  # Set the model to evaluation mode
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model, metadata, feature_definitions
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to load model from {model_path}")

def evaluate_model(model, df, feature_definitions, context_length, prediction_length, batch_size=128):
    """
    Evaluate the model on data
    
    Args:
        model: TFT model
        df: DataFrame with data
        feature_definitions: Feature definitions
        context_length: Context length
        prediction_length: Prediction length
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Dictionary of evaluation metrics
        actuals: Array of actual values
        predictions: Array of predicted values
        quantile_predictions: Array of quantile predictions if available
    """
    try:
        logger.info("Evaluating model performance...")
        
        # Use a simple dataset that provides both features and targets
        feature_cols = feature_definitions.get('temporal_real_features', 
                                              [col for col in df.columns if col != 'returns_10sec'])
        
        # Get the expected feature dimension from the model
        expected_features = model.time_varying_real_variables
        logger.info(f"Model expects {expected_features} features, dataset has {len(feature_cols)} features")
        
        # If the dimensions don't match, we need to adapt
        if len(feature_cols) != expected_features:
            logger.warning(f"Feature count mismatch. Using first {expected_features} features for evaluation.")
            # Take the first expected_features columns if we have more than needed
            if len(feature_cols) > expected_features:
                feature_cols = feature_cols[:expected_features]
            # Or pad with zeros if we have fewer than needed
            else:
                # This will be handled by padding the tensors later
                pass
        
        # Create a dataset for the entire dataframe
        dataset = DatasetForEvaluation(
            data=df,
            feature_cols=feature_cols,
            target_col='returns_10sec',  # Default target column
            context_length=context_length,
            prediction_length=prediction_length
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False
        )
        
        # Track metrics
        all_predictions = []
        all_actuals = []
        all_quantile_predictions = []
        
        # Set model to evaluation mode
        model.eval()
        
        # Evaluate in batches
        with torch.no_grad():
            # Get the first batch to inspect keys
            try:
                sample_batch = next(iter(data_loader))
                logger.info(f"Batch keys: {sample_batch.keys()}")
                
                # Check feature dimensions
                temporal_features = sample_batch['temporal_real_features']
                feature_dim = temporal_features.shape[2]
                logger.info(f"Temporal features shape: {temporal_features.shape}, feature dim: {feature_dim}")
                
                if feature_dim != expected_features:
                    logger.warning(f"Adjusting feature dimension from {feature_dim} to {expected_features}")
            except Exception as e:
                logger.error(f"Error examining batch: {str(e)}")
            
            for batch_idx, batch in enumerate(data_loader):
                # Get inputs - simpler since we're using a custom dataset
                temporal_real = batch['temporal_real_features']
                target = batch['targets']
                
                # Adjust feature dimension if needed
                if temporal_real.shape[2] != expected_features:
                    if temporal_real.shape[2] < expected_features:
                        # Pad with zeros
                        padding = torch.zeros(
                            (temporal_real.shape[0], temporal_real.shape[1], expected_features - temporal_real.shape[2]),
                            device=temporal_real.device,
                            dtype=temporal_real.dtype
                        )
                        temporal_real = torch.cat([temporal_real, padding], dim=2)
                    else:
                        # Take first expected_features columns
                        temporal_real = temporal_real[:, :, :expected_features]
                
                # Forward pass with only the temporal_real_features
                outputs = model(
                    temporal_real_features=temporal_real
                )
                
                # Get predictions
                predictions = outputs['prediction']
                
                # Handle different output formats
                if hasattr(model, 'loss_fn') and model.loss_fn == "quantile":
                    # For quantile predictions, separate by quantiles
                    point_predictions = predictions[:, -1, 1].cpu().numpy()  # Middle quantile
                    quantile_preds = predictions[:, -1, :].cpu().numpy()
                    all_quantile_predictions.append(quantile_preds)
                else:
                    # For MSE/point predictions
                    point_predictions = predictions[:, -1].cpu().numpy()  # Last timestep
                    if len(point_predictions.shape) > 1:
                        point_predictions = point_predictions.squeeze(-1)
                
                # Store predictions and actuals
                all_predictions.append(point_predictions)
                all_actuals.append(target.cpu().numpy())
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Evaluated {batch_idx + 1} batches")
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        actuals = np.concatenate(all_actuals)
        
        if all_quantile_predictions:
            quantile_predictions = np.concatenate(all_quantile_predictions)
        else:
            quantile_predictions = None
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        bias = np.mean(predictions - actuals)
        
        # Direction accuracy
        direction_accuracy = np.mean((np.sign(predictions) == np.sign(actuals)).astype(float))
        
        # Create metrics dictionary
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'bias': bias,
            'direction_accuracy': direction_accuracy
        }
        
        logger.info(f"Evaluation metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.6f}")
        logger.info(f"Bias={bias:.6f}, Direction Accuracy={direction_accuracy:.2f}")
        
        return metrics, actuals, predictions, quantile_predictions
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def plot_predictions(actuals, predictions, title="Model Predictions vs Actuals", 
                      save_path=None, sample_size=1000, plot_type='scatter'):
    """
    Create visualization of predictions vs actuals
    
    Parameters:
    actuals (np.ndarray): Actual values
    predictions (np.ndarray): Predicted values
    title (str): Plot title
    save_path (str): Path to save the plot
    sample_size (int): Number of samples to plot (to avoid overcrowding)
    plot_type (str): Type of plot ('scatter', 'time', 'histogram', 'density')
    
    Returns:
    None
    """
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Sample data if there are too many points
    if len(actuals) > sample_size:
        indices = np.random.choice(len(actuals), sample_size, replace=False)
        actuals_sample = actuals[indices]
        predictions_sample = predictions[indices]
    else:
        actuals_sample = actuals
        predictions_sample = predictions
    
    if plot_type == 'scatter':
        # Scatter plot with regression line
        plt.scatter(actuals_sample, predictions_sample, alpha=0.5, label='Predictions')
        
        # Perfect prediction line
        min_val = min(actuals_sample.min(), predictions_sample.min())
        max_val = max(actuals_sample.max(), predictions_sample.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Linear regression line
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(actuals_sample.reshape(-1, 1), predictions_sample)
        line_x = np.linspace(min_val, max_val, 100)
        line_y = model.predict(line_x.reshape(-1, 1))
        plt.plot(line_x, line_y, 'g-', label=f'Regression Line (slope={model.coef_[0]:.3f})')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        
    elif plot_type == 'time':
        # Time series plot (first n values)
        plt.plot(actuals_sample[:sample_size], label='Actual')
        plt.plot(predictions_sample[:sample_size], label='Predicted', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        
    elif plot_type == 'histogram':
        # Histogram of errors
        errors = predictions_sample - actuals_sample
        plt.hist(errors, bins=50, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        plt.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean Error: {np.mean(errors):.4f}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Prediction Errors - {title}')
        plt.legend()
        
    elif plot_type == 'density':
        # Kernel density plot of predictions vs actuals
        sns.kdeplot(actuals_sample, label='Actual')
        sns.kdeplot(predictions_sample, label='Predicted')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Distribution of Actual vs Predicted Values - {title}')
        plt.legend()
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Plot saved to {save_path}")
    
    plt.close()

def plot_feature_importance(feature_importance, feature_names, save_path=None, top_n=20):
    """
    Plot feature importance from the TFT model
    
    Parameters:
    feature_importance (np.ndarray): Feature importance scores
    feature_names (list): List of feature names
    save_path (str): Path to save the plot
    top_n (int): Number of top features to display
    
    Returns:
    None
    """
    # Ensure feature_importance is a 1D array
    if feature_importance.ndim > 1:
        feature_importance = feature_importance.mean(axis=0)
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {len(importance_df)} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()

def plot_attention_weights(attention_weights, timestamp_range, save_path=None):
    """
    Visualize attention weights from the TFT model
    
    Parameters:
    attention_weights (np.ndarray): Attention weights matrix
    timestamp_range (list): List of timestamps for labeling
    save_path (str): Path to save the plot
    
    Returns:
    None
    """
    # Handle different attention weight formats
    if attention_weights.ndim == 3:
        # Average over batch dimension if necessary
        attention_weights = attention_weights.mean(axis=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_weights, cmap='viridis')
    
    # Add timestamp labels if available
    if len(timestamp_range) == attention_weights.shape[1]:
        # Convert timestamps to readable format if they're datetime objects
        if isinstance(timestamp_range[0], pd.Timestamp):
            timestamp_labels = [ts.strftime('%H:%M:%S') for ts in timestamp_range]
        else:
            timestamp_labels = timestamp_range
        
        # Only show a subset of labels to avoid overcrowding
        if len(timestamp_labels) > 10:
            step = len(timestamp_labels) // 10
            plt.xticks(np.arange(0, len(timestamp_labels), step), 
                       [timestamp_labels[i] for i in range(0, len(timestamp_labels), step)],
                       rotation=45)
    
    plt.title('Attention Weights Heatmap')
    plt.ylabel('Output Timestep')
    plt.xlabel('Input Timestep')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Attention weights plot saved to {save_path}")
    
    plt.close()

def calculate_trading_metrics(actuals, predictions, trading_threshold=0.0):
    """
    Calculate trading performance metrics based on model predictions
    
    Parameters:
    actuals (np.ndarray): Actual values
    predictions (np.ndarray): Predicted values
    trading_threshold (float): Threshold for taking a position
    
    Returns:
    dict: Trading metrics
    """
    # Generate trading signals
    signals = np.zeros_like(predictions)
    signals[predictions > trading_threshold] = 1    # Long position
    signals[predictions < -trading_threshold] = -1  # Short position
    
    # Calculate PnL (profit/loss) for each trade
    # Assuming the actual is the future return that would be realized
    pnl = signals * actuals
    
    # Calculate various trading metrics
    total_trades = np.sum(signals != 0)
    winning_trades = np.sum((pnl > 0) & (signals != 0))
    losing_trades = np.sum((pnl < 0) & (signals != 0))
    
    # Avoid division by zero
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate returns
    total_return = np.sum(pnl)
    avg_return_per_trade = np.mean(pnl[signals != 0]) if total_trades > 0 else 0
    
    # Calculate Sharpe ratio (simplified, assuming daily returns)
    sharpe_ratio = np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0
    
    # Calculate max drawdown
    cumulative_returns = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate Calmar ratio (annualized return / max drawdown)
    # Simplified calculation assuming daily data
    annualized_return = total_return / len(actuals) * 252
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': win_rate,
        'total_return': float(total_return),
        'avg_return_per_trade': float(avg_return_per_trade),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'profit_factor': float(np.sum(pnl[pnl > 0]) / -np.sum(pnl[pnl < 0])) if np.sum(pnl[pnl < 0]) != 0 else float('inf')
    }
    
    logger.info(f"Trading metrics: {metrics}")
    return metrics

def plot_equity_curve(pnl, save_path=None, title="Trading Performance"):
    """
    Plot the equity curve from trading results
    
    Parameters:
    pnl (np.ndarray): Array of profit/loss values
    save_path (str): Path to save the plot
    title (str): Plot title
    
    Returns:
    None
    """
    equity_curve = np.cumsum(pnl)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Calculate drawdowns
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    
    # Plot drawdowns
    plt.fill_between(np.arange(len(equity_curve)), 
                    equity_curve, 
                    running_max, 
                    where=(equity_curve < running_max), 
                    color='red', 
                    alpha=0.3, 
                    label='Drawdown')
    
    plt.title(title)
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    
    # Add some stats to the plot
    total_return = equity_curve[-1]
    max_drawdown = np.max(drawdown)
    
    stats_text = f"Total Return: {total_return:.4f}\nMax Drawdown: {max_drawdown:.4f}"
    plt.figtext(0.02, 0.02, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Equity curve saved to {save_path}")
    
    plt.close()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def plot_feature_importance_from_scores(feature_scores, save_path=None, top_n=20, feature_type='combined'):
    """
    Plot feature importance from feature selection scores
    
    Parameters:
    feature_scores (dict): Feature importance scores from feature selection
    save_path (str): Path to save the plot
    top_n (int): Number of top features to display
    feature_type (str): Type of feature score to use ('mutual_info', 'f_regression', 
                        'correlation', 'combined')
    
    Returns:
    None
    """
    if not feature_scores:
        logger.warning("No feature scores available for plotting")
        return
    
    # Extract scores for the specified method
    scores = {}
    for feature, metrics in feature_scores.items():
        if feature_type in metrics:
            scores[feature] = metrics[feature_type]
    
    if not scores:
        logger.warning(f"No scores found for method {feature_type}")
        return
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': list(scores.keys()),
        'Importance': list(scores.values())
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {len(importance_df)} Features by {feature_type.capitalize()} Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close()

def evaluate_on_recent_data(model, days=7, handle_missing="interpolate", context_length=30, prediction_length=1, output_dir=None):
    """
    Evaluate the model on recent data and save results
    
    Args:
        model: TFT model
        days: Number of days of data to use
        handle_missing: How to handle missing values ("fill_zero", "fill_mean", "fill_forward", "interpolate")
        context_length: Context length
        prediction_length: Prediction length
        output_dir: Output directory for results (if None, creates a timestamped directory)
    
    Returns:
        Dictionary of results
    """
    logger.info(f"Loading {days} days of recent data for evaluation...")
    
    # Load data from DB
    df = load_data(days=days, handle_missing=handle_missing)
    
    # Add derived features with new microstructure metrics
    df = add_derived_features(df)
    
    # Prepare for evaluation
    logger.info("Preparing data for evaluation...")
    # Extract feature importance scores if we're using feature selection
    use_feature_selection = False
    try:
        # Get model metadata to see if feature selection was used during training
        metadata = None
        if hasattr(model, 'metadata'):
            metadata = model.metadata
        
        use_feature_selection = metadata and metadata.get('feature_selection', False)
    except:
        use_feature_selection = False
    
    # Prepare data with or without feature selection
    X, scalers, feature_definitions = prepare_data_for_tft(
        df, 
        context_length, 
        prediction_length,
        feature_selection=use_feature_selection,
        n_features=30,  # Default to 30 features if using selection
        selection_method='combined'
    )
    
    # Evaluate model
    metrics, actuals, predictions, quantile_predictions = evaluate_model(
        model, 
        X, 
        feature_definitions, 
        context_length, 
        prediction_length
    )
    
    # Create output directory if needed
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/evaluation/eval_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature count safely
    feature_count = 0
    if feature_definitions and 'temporal_real_cols' in feature_definitions:
        feature_count = len(feature_definitions['temporal_real_cols'])
    else:
        # Try to determine from the model
        try:
            feature_count = model.time_varying_real_variables
        except:
            feature_count = 0
    
    # Save results
    results = {
        "metrics": metrics,
        "model_info": {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "timestamp": datetime.now().isoformat()
        },
        "data_info": {
            "days": days,
            "handle_missing": handle_missing,
            "total_samples": len(actuals),
            "feature_count": feature_count
        }
    }
    
    # Sample predictions for visualization
    sample_size = min(1000, len(actuals))
    
    # Create plots
    logger.info("Creating prediction vs actual scatter plot...")
    plot_predictions(actuals, predictions, 
                    "Model Predictions vs Actuals", 
                    f"{output_dir}/predictions_scatter.png",
                    sample_size=sample_size,
                    plot_type="scatter")
    
    logger.info("Creating time series plot...")
    plot_predictions(actuals, predictions, 
                    "Predictions and Actuals Over Time", 
                    f"{output_dir}/predictions_time.png",
                    sample_size=sample_size,
                    plot_type="time")
    
    logger.info("Creating error histogram...")
    plot_predictions(actuals, predictions, 
                    "Prediction Error Distribution", 
                    f"{output_dir}/prediction_errors.png",
                    sample_size=sample_size,
                    plot_type="histogram")
    
    logger.info("Creating distribution comparison...")
    plot_predictions(actuals, predictions, 
                    "Prediction vs Actual Distributions", 
                    f"{output_dir}/prediction_distribution.png",
                    sample_size=sample_size,
                    plot_type="density")
    
    # Plot feature importance if available
    if 'feature_scores' in feature_definitions and feature_definitions['feature_scores']:
        logger.info("Creating feature importance plot...")
        plot_feature_importance_from_scores(
            feature_definitions['feature_scores'],
            save_path=f"{output_dir}/feature_importance.png",
            top_n=20,
            feature_type='combined'
        )
        
        # Save feature scores to JSON
        try:
            with open(f"{output_dir}/feature_scores.json", 'w') as f:
                # Convert any numpy values to Python types for JSON serialization
                feature_scores = feature_definitions['feature_scores']
                serializable_scores = {}
                for feature, scores in feature_scores.items():
                    serializable_scores[feature] = {
                        k: float(v) if hasattr(v, 'item') else v
                        for k, v in scores.items()
                    }
                json.dump(serializable_scores, f, indent=2)
            logger.info(f"Feature scores saved to {output_dir}/feature_scores.json")
        except Exception as e:
            logger.warning(f"Failed to save feature scores: {e}")
    
    # Calculate trading metrics with multiple thresholds
    logger.info("Calculating trading metrics with multiple thresholds...")
    thresholds = [0.0, 0.0001, 0.0002, 0.0005]
    best_threshold = 0.0
    best_sharpe = 0.0
    threshold_metrics = {}
    
    for threshold in thresholds:
        trading_metrics = calculate_trading_metrics(actuals, predictions, threshold)
        threshold_metrics[str(threshold)] = trading_metrics
        
        # Keep track of best performing threshold by Sharpe ratio
        if trading_metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = trading_metrics['sharpe_ratio']
            best_threshold = threshold
    
    results["trading_metrics"] = threshold_metrics
    results["best_threshold"] = best_threshold
    
    logger.info(f"Best threshold: {best_threshold} (Sharpe: {best_sharpe:.4f})")
    
    # Create equity curve with best threshold
    logger.info(f"Creating equity curve with threshold {best_threshold}...")
    equity_curve = create_equity_curve(actuals, predictions, threshold=best_threshold)
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f'Equity Curve (Threshold: {best_threshold})')
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.savefig(f"{output_dir}/equity_curve.png")
    plt.close()
    
    # Plot equity curves for all thresholds for comparison
    plt.figure(figsize=(12, 6))
    for threshold in thresholds:
        equity = create_equity_curve(actuals, predictions, threshold=threshold)
        plt.plot(equity, label=f'Threshold: {threshold}')
    plt.title('Equity Curves for Different Thresholds')
    plt.xlabel('Trade')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/equity_curves_comparison.png")
    plt.close()
    
    # Save results to JSON file
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Save sample data for reference
    sample_data = {
        "actuals": actuals[:sample_size].tolist(),
        "predictions": predictions[:sample_size].tolist()
    }
    
    with open(f"{output_dir}/sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return results

def create_equity_curve(actuals, predictions, threshold=0.0):
    """
    Create equity curve from predictions and actuals
    
    Args:
        actuals: Array of actual values
        predictions: Array of predictions
        threshold: Threshold for trading (only trade if abs(prediction) > threshold)
        
    Returns:
        Array of cumulative returns
    """
    # Simple strategy: direction of prediction
    signals = np.sign(predictions)
    signals[np.abs(predictions) <= threshold] = 0  # No trade if prediction is below threshold
    
    # Calculate returns
    returns = signals * actuals
    
    # Cumulative returns
    equity_curve = np.cumsum(returns)
    
    return equity_curve

def main():
    parser = argparse.ArgumentParser(description='Evaluate improved TFT model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--days', type=int, default=7, help='Number of days of recent data to evaluate on')
    parser.add_argument('--missing', type=str, default='interpolate', 
                        choices=['fill_zero', 'fill_mean', 'fill_forward', 'interpolate'], 
                        help='Method for handling missing values')
    parser.add_argument('--output', type=str, default=None, help='Output directory for evaluation results')
    parser.add_argument('--context', type=int, default=None, help='Context length override (default: use model config)')
    parser.add_argument('--feature-selection', action='store_true', help='Apply feature selection during evaluation')
    
    args = parser.parse_args()
    
    # Load model
    model, metadata, feature_definitions = load_model(args.model)
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Get context length from model or args
    context_length = args.context if args.context is not None else model.hparams.context_length
    
    # Evaluate on recent data
    evaluate_on_recent_data(
        model=model,
        days=args.days,
        handle_missing=args.missing,
        context_length=context_length,
        prediction_length=model.hparams.prediction_length,
        output_dir=args.output
    )
    
if __name__ == "__main__":
    main() 