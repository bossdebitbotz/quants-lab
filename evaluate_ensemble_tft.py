import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
from train_improved_tft_with_better_targets import DirectionalTFT, load_data
from train_downward_specialist_tft import DownwardSpecialistTFT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ensemble_eval")

def load_model(model_path, model_type="directional"):
    """Load a trained model from file"""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Fix for PyTorch 2.6: Use weights_only=False explicitly
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        config = checkpoint.get('config', None)
        
        if config is None:
            logger.error(f"No configuration found in model file: {model_path}")
            return None
        
        if model_type == "directional":
            model = DirectionalTFT(**config)
        elif model_type == "downward":
            model = DownwardSpecialistTFT(**config)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model {model_type} loaded from {model_path}")
        
        # Extract expected feature count
        feature_count = config.get('time_varying_real_variables', None)
        logger.info(f"Model expects {feature_count} input features")
        
        return model, feature_count
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None, None

def prepare_evaluation_data(df, target_col, context_length=20):
    """Prepare data for model evaluation"""
    # Select features (all columns except targets)
    feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'timestamp' and col != 'original_target']
    
    # Create sequences for evaluation
    sequences = []
    targets = []
    timestamps = []
    
    for i in range(context_length, len(df)):
        # Get context window
        sequence = df[feature_cols].iloc[i-context_length:i].values
        # Get target
        target = df[target_col].iloc[i]
        # Get timestamp if available
        timestamp = df['timestamp'].iloc[i] if 'timestamp' in df.columns else None
        
        sequences.append(sequence)
        targets.append(target)
        if timestamp is not None:
            timestamps.append(timestamp)
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    logger.info(f"Prepared {len(sequences)} evaluation samples with context length {context_length}")
    logger.info(f"Input feature count: {len(feature_cols)}")
    
    return sequences, targets, timestamps, feature_cols

def adapt_features_to_model(sequences, expected_feature_count):
    """
    Adapt the feature count to match model expectations
    
    Args:
        sequences: Input sequences with shape [batch, seq_len, feature_count]
        expected_feature_count: Expected number of features for the model
        
    Returns:
        Adapted sequences
    """
    current_feature_count = sequences.shape[2]
    
    if current_feature_count == expected_feature_count:
        return sequences
    
    logger.info(f"Adapting features from {current_feature_count} to {expected_feature_count}")
    
    if current_feature_count < expected_feature_count:
        # Pad with zeros
        padding_size = expected_feature_count - current_feature_count
        batch_size, seq_len, _ = sequences.shape
        padding = np.zeros((batch_size, seq_len, padding_size))
        adapted_sequences = np.concatenate([sequences, padding], axis=2)
        logger.info(f"Padded input with {padding_size} zero features")
        return adapted_sequences
    else:
        # Truncate
        logger.warning(f"Truncating {current_feature_count - expected_feature_count} features")
        return sequences[:, :, :expected_feature_count]

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Direction accuracy
    pred_sign = np.sign(predictions)
    true_sign = np.sign(targets)
    direction_acc = np.mean(pred_sign == true_sign)
    
    # Accuracy by class
    pos_mask = (true_sign > 0)
    neg_mask = (true_sign < 0)
    zero_mask = (true_sign == 0)
    
    pos_acc = np.mean(pred_sign[pos_mask] == true_sign[pos_mask]) if np.any(pos_mask) else 0
    neg_acc = np.mean(pred_sign[neg_mask] == true_sign[neg_mask]) if np.any(neg_mask) else 0
    zero_acc = np.mean(pred_sign[zero_mask] == true_sign[zero_mask]) if np.any(zero_mask) else 0
    
    # Balance (up/down ratio in predictions)
    pred_up = np.sum(pred_sign > 0)
    pred_down = np.sum(pred_sign < 0)
    balance = pred_up / pred_down if pred_down > 0 else float('inf')
    
    # Bias
    bias = np.mean(predictions - targets)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_acc,
        'positive_accuracy': pos_acc,
        'negative_accuracy': neg_acc,
        'zero_accuracy': zero_acc,
        'balance': balance,
        'bias': bias
    }
    
    return metrics

def simulate_trading(predictions, targets, threshold=0.0001, output_dir=None, 
                   position_sizing=True, stop_loss=0.003, take_profit=0.006, 
                   max_holding_period=10, use_dynamic_threshold=True,
                   fee_per_trade=0.0001):
    """Simulate trading based on predictions with enhanced risk management
    
    Args:
        predictions: Model predictions
        targets: Actual target values
        threshold: Base threshold for trade entry
        output_dir: Directory to save plots
        position_sizing: Whether to use adaptive position sizing based on prediction strength
        stop_loss: Stop loss as a percentage of position value
        take_profit: Take profit as a percentage of position value
        max_holding_period: Maximum number of periods to hold a position
        use_dynamic_threshold: Whether to use dynamic thresholds based on recent volatility
        fee_per_trade: Transaction cost per trade (one-way)
    """
    # Ensure predictions and targets are 1D arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Print shapes for debugging
    logger.info(f"Shapes - predictions: {predictions.shape}, targets: {targets.shape}")
    
    # Initialize variables
    prediction_signs = np.sign(predictions)
    true_signs = np.sign(targets)
    
    # Calculate dynamic threshold if enabled
    if use_dynamic_threshold:
        # Use rolling standard deviation of targets to adjust threshold
        rolling_window = min(30, len(targets) // 10)  # Use at least 10% of data
        rolling_std = np.zeros_like(targets)
        
        for i in range(len(targets)):
            start_idx = max(0, i - rolling_window)
            rolling_std[i] = np.std(targets[start_idx:i+1]) if i > 5 else np.std(targets[:i+1])
        
        # Scale threshold by volatility (higher volatility = higher threshold)
        mean_std = np.mean(rolling_std) if np.mean(rolling_std) > 0 else 1.0
        dynamic_threshold = threshold * (1 + rolling_std / mean_std)
        entry_threshold = dynamic_threshold
    else:
        entry_threshold = np.ones_like(predictions) * threshold
    
    # Initialize trade tracking
    positions = np.zeros_like(predictions)  # 0: no position, 1: long, -1: short
    position_sizes = np.zeros_like(predictions)  # Fraction of capital
    entry_prices = np.zeros_like(predictions)
    entry_indices = np.zeros_like(predictions, dtype=int) - 1  # -1 means no entry
    holding_periods = np.zeros_like(predictions, dtype=int)
    
    # Capital tracking
    initial_capital = 1.0
    capital = np.ones_like(predictions) * initial_capital
    
    # For return tracking
    trade_returns = []
    trade_durations = []
    trade_directions = []
    
    # Simulate trading
    for i in range(1, len(predictions)):
        # Update holding period for existing positions
        if positions[i-1] != 0:
            holding_periods[i] = holding_periods[i-1] + 1
            entry_indices[i] = entry_indices[i-1]  # Carry forward entry index
        
        # Check for exits (stop loss, take profit, max holding)
        if positions[i-1] != 0:
            # Calculate current P&L as percentage
            if i > 0 and entry_indices[i-1] >= 0:
                # Calculate the return since entry
                current_pnl = positions[i-1] * (targets[i] if i > 0 else 0.0)
                
                # Exit conditions
                exit_position = False
                
                # Stop loss hit
                if current_pnl < -stop_loss:
                    exit_position = True
                    exit_reason = "stop_loss"
                
                # Take profit hit
                elif current_pnl > take_profit:
                    exit_position = True
                    exit_reason = "take_profit"
                
                # Max holding period reached
                elif holding_periods[i] >= max_holding_period:
                    exit_position = True
                    exit_reason = "max_holding"
                
                # Exit logic
                if exit_position:
                    # Calculate full trade return from entry to exit
                    idx_since_entry = range(entry_indices[i-1] + 1, i + 1)
                    trade_pnl = positions[i-1] * sum(targets[idx] for idx in idx_since_entry)
                    
                    # Record trade
                    trade_returns.append(trade_pnl)
                    trade_durations.append(holding_periods[i-1])
                    trade_directions.append(positions[i-1])
                    
                    # Update capital (deduct fees)
                    capital[i] = capital[i-1] * (1 + trade_pnl - fee_per_trade)
                    
                    # Reset position
                    positions[i] = 0
                    position_sizes[i] = 0
                    holding_periods[i] = 0
                    entry_indices[i] = -1
                    continue  # Skip to next iteration
            
            # If no exit, carry forward position
            positions[i] = positions[i-1]
            position_sizes[i] = position_sizes[i-1]
            capital[i] = capital[i-1] * (1 + positions[i-1] * targets[i])  # Apply daily P&L
            continue
        
        # Entry logic (only if not already in a position)
        if positions[i-1] == 0:
            # Long entry
            if predictions[i] > entry_threshold[i]:
                positions[i] = 1
                
                # Position sizing
                if position_sizing:
                    # Scale position size by prediction strength
                    strength = min(1.0, predictions[i] / (entry_threshold[i] * 5))
                    position_sizes[i] = 0.1 + 0.4 * strength  # 10-50% of capital
                else:
                    position_sizes[i] = 0.3  # Fixed 30% allocation
                
                # Record entry
                entry_indices[i] = i
                
                # Reset holding period
                holding_periods[i] = 1
                
                # Deduct fees from capital
                capital[i] = capital[i-1] * (1 - fee_per_trade)
                
            # Short entry    
            elif predictions[i] < -entry_threshold[i]:
                positions[i] = -1
                
                # Position sizing
                if position_sizing:
                    # Scale position size by prediction strength
                    strength = min(1.0, abs(predictions[i]) / (entry_threshold[i] * 5))
                    position_sizes[i] = 0.1 + 0.4 * strength  # 10-50% of capital
                else:
                    position_sizes[i] = 0.3  # Fixed 30% allocation
                
                # Record entry
                entry_indices[i] = i
                
                # Reset holding period
                holding_periods[i] = 1
                
                # Deduct fees from capital
                capital[i] = capital[i-1] * (1 - fee_per_trade)
                
            else:
                # No new position, carry forward capital
                capital[i] = capital[i-1]
    
    # Calculate final P&L for any open positions
    for i in range(len(predictions)):
        if positions[i] != 0 and i == len(predictions) - 1 and entry_indices[i] >= 0:
            # Calculate full trade return from entry to last bar
            idx_since_entry = range(entry_indices[i] + 1, i + 1)
            trade_pnl = positions[i] * sum(targets[idx] for idx in idx_since_entry if idx < len(targets))
            
            trade_returns.append(trade_pnl)
            trade_durations.append(holding_periods[i])
            trade_directions.append(positions[i])
    
    # Calculate metrics
    total_trades = len(trade_returns)
    winning_trades = sum(np.array(trade_returns) > 0) if total_trades > 0 else 0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate final equity
    final_equity = capital[-1]
    total_return = final_equity - initial_capital
    
    # Calculate Sharpe ratio (approximation)
    # Assuming 252 trading days per year and scaling to that timeframe
    daily_returns = np.diff(np.log(capital))
    daily_returns = daily_returns[~np.isnan(daily_returns) & ~np.isinf(daily_returns)]
    
    if len(daily_returns) > 1:
        returns_std = np.std(daily_returns)
        returns_mean = np.mean(daily_returns)
        if returns_std > 0:
            sharpe = (returns_mean / returns_std) * np.sqrt(252 * 24 * 60 / 10)  # Assuming 10-second bars
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Average return per trade
    avg_return_per_trade = np.mean(trade_returns) if total_trades > 0 else 0
    
    # Print results
    metrics = {
        'threshold': threshold,
        'total_trades': int(total_trades),
        'win_rate': float(win_rate) if total_trades > 0 else 0,
        'total_return': float(total_return),
        'avg_return_per_trade': float(avg_return_per_trade),
        'sharpe_ratio': float(sharpe),
        'avg_trade_duration': float(np.mean(trade_durations)) if trade_durations else 0,
        'final_equity': float(final_equity)
    }
    
    # Generate plots if output directory specified
    if output_dir and total_trades > 0:
        # Plot equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(capital, label=f'Threshold={threshold}')
        plt.title(f'Equity Curve (Threshold={threshold})')
        plt.xlabel('Bar #')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/equity_curve_t{threshold}.png")
        plt.close()
        
        # Plot trade returns histogram if we have enough trades
        if total_trades >= 10:
            plt.figure(figsize=(10, 6))
            plt.hist(trade_returns, bins=min(50, total_trades//2))
            plt.title(f'Trade Returns Distribution (Threshold={threshold})')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f"{output_dir}/trade_returns_t{threshold}.png")
            plt.close()
            
            # Plot trade duration histogram
            plt.figure(figsize=(10, 6))
            plt.hist(trade_durations, bins=min(20, total_trades//5))
            plt.title(f'Trade Duration Distribution (Threshold={threshold})')
            plt.xlabel('Duration (bars)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(f"{output_dir}/trade_durations_t{threshold}.png")
            plt.close()
    
    return metrics

def ensemble_predictions(dir_predictions, down_predictions, ensemble_method='adaptive', threshold=0.003):
    """Combine predictions from directional and downward models with improvements

    Args:
        dir_predictions: Predictions from directional model
        down_predictions: Predictions from downward specialist model
        ensemble_method: How to combine predictions ('adaptive', 'average', 'max_abs', 'selective')
        threshold: Cutoff for using downward specialist in adaptive method
    
    Returns:
        Combined predictions
    """
    if ensemble_method == 'average':
        # Simple average
        return (dir_predictions + down_predictions) / 2
    
    elif ensemble_method == 'max_abs':
        # Take prediction with larger absolute value
        result = np.zeros_like(dir_predictions)
        for i in range(len(dir_predictions)):
            if abs(dir_predictions[i]) > abs(down_predictions[i]):
                result[i] = dir_predictions[i]
            else:
                result[i] = down_predictions[i]
        return result
    
    elif ensemble_method == 'adaptive':
        # Use downward specialist for predicted downward movements, directional for upward
        result = np.copy(dir_predictions)
        down_mask = (down_predictions < -threshold)
        result[down_mask] = down_predictions[down_mask]
        return result
    
    elif ensemble_method == 'selective':
        # More sophisticated selection based on confidence
        result = np.zeros_like(dir_predictions)
        
        # For each prediction, select the most confident one
        # but with bias correction based on historical performance
        dir_up_conf = np.abs(dir_predictions) * (dir_predictions > 0)
        dir_down_conf = np.abs(dir_predictions) * (dir_predictions < 0)
        down_conf = np.abs(down_predictions) * (down_predictions < 0)
        
        # Directional model is better at upward, so use it for positive predictions
        up_mask = (dir_up_conf > 0)
        result[up_mask] = dir_predictions[up_mask]
        
        # Downward model is better at downward, so use it for negative predictions
        down_mask = (down_conf > 0) & ~up_mask  # Only where not already using directional
        result[down_mask] = down_predictions[down_mask]
        
        # For remaining predictions, use directional model (as fallback)
        remaining = ~(up_mask | down_mask)
        result[remaining] = dir_predictions[remaining]
        
        return result
    
    else:
        logger.warning(f"Unknown ensemble method: {ensemble_method}, using average")
        return (dir_predictions + down_predictions) / 2

def plot_confusion(true_signs, pred_signs, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_signs, pred_signs, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Down', 'Zero', 'Up'],
                yticklabels=['Down', 'Zero', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

def plot_accuracy_by_magnitude(targets, predictions, output_dir):
    """Plot accuracy as a function of target magnitude"""
    # Convert to numpy arrays if they're not already
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # Calculate absolute magnitude of targets
    magnitudes = np.abs(targets)
    
    # Create magnitude bins
    percentiles = np.percentile(magnitudes[magnitudes > 0], 
                               [25, 50, 75, 90, 95, 99])
    bins = [0] + list(percentiles)
    bin_labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-95%', '95-99%', '99-100%']
    
    # Calculate bin indices
    bin_indices = np.digitize(magnitudes, bins)
    
    # Calculate direction accuracy by bin
    accuracies = []
    counts = []
    
    for i in range(1, len(bins) + 1):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            acc = np.mean(np.sign(predictions[mask]) == np.sign(targets[mask]))
            count = np.sum(mask)
            accuracies.append(acc)
            counts.append(count)
        else:
            accuracies.append(0)
            counts.append(0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart with counts on secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars = ax1.bar(bin_labels[:len(accuracies)], accuracies, alpha=0.7, color='blue')
    ax1.set_xlabel('Target Magnitude Percentile')
    ax1.set_ylabel('Direction Accuracy')
    ax1.set_ylim([0, 1])
    
    # Add count line
    line = ax2.plot(bin_labels[:len(counts)], counts, 'ro-', label='Count')
    ax2.set_ylabel('Sample Count')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Direction Accuracy by Target Magnitude')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_magnitude.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of TFT models')
    parser.add_argument('--dir_model', type=str, required=True,
                        help='Path to directional model file')
    parser.add_argument('--down_model', type=str, required=True,
                        help='Path to downward specialist model file')
    parser.add_argument('--dataset', type=str, default='data_improvements/improved_dataset_small.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--target', type=str, default='target_ma_adjusted',
                       help='Target column to evaluate against')
    parser.add_argument('--context', type=int, default=20,
                       help='Context length (time steps)')
    parser.add_argument('--ensemble', type=str, default='adaptive',
                       choices=['adaptive', 'average', 'max_abs', 'selective'],
                       help='Ensemble method to combine predictions')
    parser.add_argument('--threshold', type=float, default=0.003,
                       help='Threshold for adaptive ensemble method')
    parser.add_argument('--position_sizing', action='store_true',
                       help='Enable position sizing based on prediction strength')
    parser.add_argument('--stop_loss', type=float, default=0.003,
                       help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=0.006,
                       help='Take profit percentage')
    parser.add_argument('--max_holding', type=int, default=10,
                       help='Maximum holding period (in bars)')
    parser.add_argument('--use_dynamic_threshold', action='store_true',
                       help='Use dynamic thresholds based on volatility')
    parser.add_argument('--fee', type=float, default=0.0001,
                       help='Fee per trade (one-way)')
    
    args = parser.parse_args()
    
    # Load models
    dir_model, dir_features = load_model(args.dir_model, model_type="directional")
    down_model, down_features = load_model(args.down_model, model_type="downward")
    
    if dir_model is None or down_model is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"experiments/evaluation/ensemble_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    df = load_data(args.dataset)
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    sequences, targets, timestamps, feature_cols = prepare_evaluation_data(
        df, args.target, context_length=args.context
    )
    
    # Make predictions with both models
    logger.info("Making predictions with directional model...")
    dir_predictions = []
    
    # Adapt sequences for directional model
    if dir_features:
        dir_sequences = adapt_features_to_model(sequences, dir_features)
    else:
        dir_sequences = sequences
    
    with torch.no_grad():
        for i in range(0, len(dir_sequences), 32):  # Process in batches
            batch = dir_sequences[i:i+32]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            
            outputs = dir_model.forward(temporal_real_features=batch_tensor)
            batch_preds = outputs['prediction'].squeeze(-1).numpy()
            dir_predictions.extend(batch_preds)
    
    dir_predictions = np.array(dir_predictions)
    
    logger.info("Making predictions with downward specialist model...")
    down_predictions = []
    
    # Adapt sequences for downward model
    if down_features:
        down_sequences = adapt_features_to_model(sequences, down_features)
    else:
        down_sequences = sequences
    
    with torch.no_grad():
        for i in range(0, len(down_sequences), 32):  # Process in batches
            batch = down_sequences[i:i+32]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            
            outputs = down_model.forward(temporal_real_features=batch_tensor)
            batch_preds = outputs['prediction'].squeeze(-1).numpy()
            down_predictions.extend(batch_preds)
    
    down_predictions = np.array(down_predictions)
    
    # Combine predictions
    logger.info(f"Combining predictions using {args.ensemble} method...")
    ensemble_preds = ensemble_predictions(
        dir_predictions, down_predictions, 
        ensemble_method=args.ensemble, 
        threshold=args.threshold
    )
    
    # Calculate metrics for all models
    metrics = {
        'directional': calculate_metrics(dir_predictions, targets),
        'downward': calculate_metrics(down_predictions, targets),
        'ensemble': calculate_metrics(ensemble_preds, targets)
    }
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        logger.info(f"--- {model_name.upper()} MODEL METRICS ---")
        logger.info(f"MAE: {model_metrics['mae']:.6f}")
        logger.info(f"RMSE: {model_metrics['rmse']:.6f}")
        logger.info(f"Direction Accuracy: {model_metrics['direction_accuracy']:.4f}")
        logger.info(f"Accuracy (Up): {model_metrics['positive_accuracy']:.4f}")
        logger.info(f"Accuracy (Down): {model_metrics['negative_accuracy']:.4f}")
        logger.info(f"Prediction Balance (Up/Down): {model_metrics['balance']:.2f}")
        logger.info(f"Bias: {model_metrics['bias']:.6f}")
    
    # Plot confusion matrix
    plot_confusion(
        np.sign(targets).astype(int) + 1,  # Convert -1,0,1 to 0,1,2
        np.sign(ensemble_preds).astype(int) + 1,
        output_dir
    )
    
    # Plot accuracy by magnitude
    plot_accuracy_by_magnitude(targets, ensemble_preds, output_dir)
    
    # Simulate trading with different thresholds
    trading_thresholds = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]
    trading_results = {}
    
    # Find best threshold
    best_sharpe = 0
    best_threshold = 0
    
    for threshold in trading_thresholds:
        logger.info(f"Simulating trading with threshold {threshold}...")
        results = simulate_trading(
            ensemble_preds, targets, threshold, output_dir,
            position_sizing=args.position_sizing,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_holding_period=args.max_holding,
            use_dynamic_threshold=args.use_dynamic_threshold,
            fee_per_trade=args.fee
        )
        trading_results[str(threshold)] = results
        
        logger.info(f"  Trades: {results['total_trades']}, Win Rate: {results['win_rate']:.4f}, "
                   f"Return: {results['total_return']:.6f}, Sharpe: {results['sharpe_ratio']:.4f}")
        
        if results['sharpe_ratio'] > best_sharpe and results['total_trades'] >= 10:
            best_sharpe = results['sharpe_ratio']
            best_threshold = threshold
    
    if best_sharpe > 0:
        logger.info(f"Best threshold: {best_threshold} with Sharpe ratio: {best_sharpe:.4f}")
    else:
        logger.info("No profitable threshold found")
    
    # Save evaluation results
    results = {
        'models': {
            'directional': args.dir_model,
            'downward': args.down_model
        },
        'dataset': args.dataset,
        'target': args.target,
        'samples': len(sequences),
        'ensemble_method': args.ensemble,
        'ensemble_threshold': args.threshold,
        'metrics': metrics,
        'trading_results': trading_results,
        'best_threshold': best_threshold,
        'best_sharpe': best_sharpe,
        'confusion_matrix': confusion_matrix(
            np.sign(targets).astype(int) + 1,
            np.sign(ensemble_preds).astype(int) + 1,
            normalize='true'
        ).tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")

if __name__ == "__main__":
    main()