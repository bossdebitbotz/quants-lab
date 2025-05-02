import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
import json
from torch.serialization import add_safe_globals

# Import our model and dataset class
from train_improved_tft_with_better_targets import DirectionalTFT, ImprovedDataset, load_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_evaluation")

# Add safe globals for PyTorch Lightning
add_safe_globals(['lightning_fabric.utilities.data.AttributeDict'])

def prepare_evaluation_data(df, feature_cols, target_col='target_sign'):
    """Prepare data for evaluation"""
    # Create dataset
    dataset = ImprovedDataset(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        context_length=10,  # Use same as training
        prediction_length=1
    )
    
    # Create dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    return data_loader, dataset

def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model on provided data"""
    model.to(device)
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Extract data from batch
            temporal_real = batch['temporal_real_features'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            outputs = model.forward(
                temporal_real_features=temporal_real
            )
            
            # Process predictions
            predictions = outputs['prediction'].squeeze(-1)
            
            # Store predictions and targets
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            # Calculate probability-like values for ROC curve
            # For sign prediction, use raw values before sign as "probabilities"
            all_probabilities.append(predictions.cpu().numpy())
    
    # Combine batches
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # For directional models, calculate direction accuracy
    pred_signs = np.sign(predictions)
    true_signs = np.sign(targets)
    
    # Handle zeros (no change) - if prediction is 0, count as wrong
    # unless true sign is also 0
    zero_preds = pred_signs == 0
    pred_signs[zero_preds & (true_signs != 0)] = -true_signs[zero_preds & (true_signs != 0)]
    
    direction_acc = np.mean(pred_signs == true_signs)
    
    # Calculate class-wise accuracy
    pos_targets = true_signs > 0
    neg_targets = true_signs < 0
    zero_targets = true_signs == 0
    
    pos_acc = np.mean(pred_signs[pos_targets] == true_signs[pos_targets]) if np.sum(pos_targets) > 0 else 0
    neg_acc = np.mean(pred_signs[neg_targets] == true_signs[neg_targets]) if np.sum(neg_targets) > 0 else 0
    zero_acc = np.mean(pred_signs[zero_targets] == true_signs[zero_targets]) if np.sum(zero_targets) > 0 else 0
    
    # Calculate confusion matrix for sign prediction
    cm = confusion_matrix(true_signs, pred_signs, labels=[-1, 0, 1])
    
    # Balance - ratio of positive to negative predictions
    balance = np.sum(pred_signs > 0) / max(1, np.sum(pred_signs < 0))
    
    # Bias (average prediction error)
    bias = np.mean(predictions - targets)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_acc,
        'pos_accuracy': pos_acc,
        'neg_accuracy': neg_acc,
        'zero_accuracy': zero_acc,
        'confusion_matrix': cm,
        'balance': balance,
        'bias': bias,
        'targets': targets,
        'predictions': predictions,
        'probabilities': probabilities,
        'pred_signs': pred_signs,
        'true_signs': true_signs,
    }

def plot_confusion_matrix(results, output_dir):
    """Plot confusion matrix for sign prediction"""
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'No Change', 'Up'],
                yticklabels=['Down', 'No Change', 'Up'])
    plt.xlabel('Predicted Direction')
    plt.ylabel('True Direction')
    plt.title('Direction Prediction Confusion Matrix')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_predictions_scatter(results, output_dir):
    """Plot scatter of predictions vs targets"""
    targets = results['targets']
    predictions = results['predictions']
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with alpha for density
    plt.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Add diagonal reference line
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'predictions_scatter.png'))
    plt.close()

def plot_sign_accuracy_by_magnitude(results, output_dir):
    """Plot how direction accuracy varies with the magnitude of true change"""
    true_signs = results['true_signs']
    pred_signs = results['pred_signs']
    targets = results['targets']
    
    # Create bins of target magnitude
    bins = 10
    target_abs = np.abs(targets)
    bin_edges = np.percentile(target_abs, np.linspace(0, 100, bins+1))
    
    # Initialize arrays to store accuracy for each bin
    bin_accuracies = []
    bin_centers = []
    bin_counts = []
    
    # Calculate accuracy within each bin
    for i in range(bins):
        if i < bins - 1:
            mask = (target_abs >= bin_edges[i]) & (target_abs < bin_edges[i+1])
        else:
            mask = target_abs >= bin_edges[i]
            
        if np.sum(mask) > 0:
            bin_acc = np.mean(pred_signs[mask] == true_signs[mask])
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2 if i < bins - 1 else bin_edges[i]
            bin_accuracies.append(bin_acc)
            bin_centers.append(bin_center)
            bin_counts.append(np.sum(mask))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(bin_accuracies)), bin_accuracies, width=0.7)
    plt.xticks(np.arange(len(bin_accuracies)), [f"{c:.3f}" for c in bin_centers], rotation=45)
    plt.xlabel('Target Magnitude Bins')
    plt.ylabel('Direction Accuracy')
    plt.title('Direction Accuracy by Target Magnitude')
    
    # Add count labels
    for i, (acc, count) in enumerate(zip(bin_accuracies, bin_counts)):
        plt.text(i, acc + 0.02, f"n={count}", ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'accuracy_by_magnitude.png'))
    plt.close()

def simulate_trading(results, output_dir, thresholds=[0.0001, 0.0005, 0.001]):
    """Simulate trading based on model predictions"""
    predictions = results['predictions']
    targets = results['targets']
    
    for threshold in thresholds:
        # Only trade when predicted value exceeds threshold
        long_signals = predictions > threshold
        short_signals = predictions < -threshold
        
        # Calculate actual returns for each position
        long_returns = targets[long_signals] if np.sum(long_signals) > 0 else np.array([])
        short_returns = -targets[short_signals] if np.sum(short_signals) > 0 else np.array([])
        
        # Combine all trade returns
        all_trade_returns = np.concatenate([long_returns, short_returns])
        
        if len(all_trade_returns) > 0:
            # Calculate trading metrics
            total_trades = len(all_trade_returns)
            win_rate = np.mean(all_trade_returns > 0)
            total_return = np.sum(all_trade_returns)
            avg_return = np.mean(all_trade_returns)
            sharpe = np.mean(all_trade_returns) / (np.std(all_trade_returns) + 1e-10) * np.sqrt(252*24*60*6)  # Annualized, assuming 10-sec data
            
            logger.info(f"Threshold: {threshold}")
            logger.info(f"Total trades: {total_trades}")
            logger.info(f"Win rate: {win_rate:.2f}")
            logger.info(f"Total return: {total_return:.4f}")
            logger.info(f"Average return per trade: {avg_return:.6f}")
            logger.info(f"Annualized Sharpe ratio: {sharpe:.4f}")
            
            # Plot return distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(all_trade_returns, bins=50, kde=True)
            plt.axvline(0, color='r', linestyle='--')
            plt.xlabel('Trade Return')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Trade Returns (Threshold = {threshold})')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'trade_returns_t{threshold}.png'))
            plt.close()
            
            # Plot cumulative returns
            plt.figure(figsize=(10, 6))
            cum_returns = np.cumsum(all_trade_returns)
            plt.plot(cum_returns)
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative Return')
            plt.title(f'Cumulative Returns (Threshold = {threshold})')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, f'equity_curve_t{threshold}.png'))
            plt.close()
        else:
            logger.info(f"No trades executed with threshold {threshold}")

    # Find best threshold
    best_threshold = None
    best_sharpe = -float('inf')
    thresholds_fine = np.linspace(0, 0.01, 100)
    sharpe_values = []
    
    for threshold in thresholds_fine:
        long_signals = predictions > threshold
        short_signals = predictions < -threshold
        
        long_returns = targets[long_signals] if np.sum(long_signals) > 0 else np.array([])
        short_returns = -targets[short_signals] if np.sum(short_signals) > 0 else np.array([])
        
        all_trade_returns = np.concatenate([long_returns, short_returns])
        
        if len(all_trade_returns) > 10:  # Require at least 10 trades
            sharpe = np.mean(all_trade_returns) / (np.std(all_trade_returns) + 1e-10) * np.sqrt(252*24*60*6)
            sharpe_values.append(sharpe)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold
        else:
            sharpe_values.append(0)
    
    if best_threshold is not None:
        logger.info(f"Best threshold: {best_threshold:.6f} with Sharpe ratio {best_sharpe:.4f}")
        
        # Plot threshold vs Sharpe
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_fine, sharpe_values)
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.6f}')
        plt.xlabel('Threshold')
        plt.ylabel('Sharpe Ratio')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'threshold_optimization.png'))
        plt.close()
    else:
        logger.warning("Could not find optimal threshold")

def load_model(model_path):
    """Load a trained TFT model"""
    try:
        # Load with weights_only=False since we need the full model structure
        saved_model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        config = saved_model.get('config', {})
        
        # Initialize model architecture
        model = DirectionalTFT.load_from_checkpoint(model_path) if model_path.endswith('.ckpt') else None
        
        if model is None:
            # For .pt files, we need to reconstruct the model from saved state
            metadata = saved_model.get('metadata', {})
            hyperparams = metadata.get('hyperparameters', {})
            
            model = DirectionalTFT(
                static_variables=[],  # No static variables
                time_varying_categorical_variables=[],  # No categorical variables
                time_varying_real_variables=metadata.get('num_features', config.get('time_varying_real_variables', 1)),
                static_embedding_sizes=[],  # No static embeddings
                time_varying_embedding_sizes=[],  # No categorical embeddings
                hidden_size=hyperparams.get('hidden_size', 64),
                lstm_layers=2,
                num_attention_heads=hyperparams.get('num_attention_heads', 2),
                dropout=hyperparams.get('dropout', 0.2),
                learning_rate=hyperparams.get('learning_rate', 5e-4),
                context_length=hyperparams.get('context_length', 10),
                prediction_length=1,
                loss_fn="directional",
                alpha=hyperparams.get('alpha', 0.5),
                bias_correction=True
            )
            
            # Load state dict
            if 'model_state_dict' in saved_model:
                model.load_state_dict(saved_model['model_state_dict'])
            else:
                logger.warning("Model state dict not found in saved model. Using un-trained model.")
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_on_dataset(model_path, dataset_path, target_col='target_sign', output_dir=None):
    """Evaluate a trained model on the specified dataset"""
    # Load model
    model = load_model(model_path)
    
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/evaluation/directional_eval_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model metadata if available
    model_metadata = None
    try:
        saved_model = torch.load(model_path, map_location=torch.device('cpu'))
        model_metadata = saved_model.get('metadata', None)
    except:
        logger.warning("Could not extract metadata from model file")
    
    # Load dataset
    df = load_data(dataset_path)
    if df is None:
        logger.error(f"Failed to load data from {dataset_path}")
        return
    
    # Determine feature columns (exclude target columns)
    target_columns = ['original_target', 'target_sign', 'target_binned', 'target_log', 'target_ma_adjusted']
    feature_cols = [col for col in df.columns if col not in target_columns]
    
    # Prepare data for evaluation
    data_loader, dataset = prepare_evaluation_data(df, feature_cols, target_col)
    
    # Evaluate model
    logger.info(f"Evaluating model on {len(dataset)} samples from {dataset_path}")
    results = evaluate_model(model, data_loader)
    
    # Print metrics
    logger.info(f"MAE: {results['mae']:.6f}")
    logger.info(f"RMSE: {results['rmse']:.6f}")
    logger.info(f"Direction Accuracy: {results['direction_accuracy']:.2f}")
    logger.info(f"Accuracy (Up): {results['pos_accuracy']:.2f}")
    logger.info(f"Accuracy (Down): {results['neg_accuracy']:.2f}")
    logger.info(f"Accuracy (No Change): {results['zero_accuracy']:.2f}")
    logger.info(f"Prediction Balance (Up/Down): {results['balance']:.2f}")
    logger.info(f"Bias: {results['bias']:.6f}")
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    plot_confusion_matrix(results, output_dir)
    plot_predictions_scatter(results, output_dir)
    plot_sign_accuracy_by_magnitude(results, output_dir)
    
    # Simulate trading
    logger.info("Simulating trading with different thresholds...")
    simulate_trading(results, output_dir)
    
    # Save results
    results_to_save = {
        'model_path': model_path,
        'dataset_path': dataset_path,
        'target_column': target_col,
        'num_samples': len(dataset),
        'metrics': {
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'direction_accuracy': float(results['direction_accuracy']),
            'pos_accuracy': float(results['pos_accuracy']),
            'neg_accuracy': float(results['neg_accuracy']),
            'zero_accuracy': float(results['zero_accuracy']),
            'balance': float(results['balance']),
            'bias': float(results['bias']),
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'model_metadata': model_metadata,
        'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate directional TFT model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--dataset', type=str, default='data_improvements/improved_dataset_small.csv', help='Path to dataset for evaluation')
    parser.add_argument('--target', type=str, default='target_sign', help='Target column to use for evaluation')
    parser.add_argument('--output', type=str, default=None, help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_on_dataset(
        model_path=args.model,
        dataset_path=args.dataset,
        target_col=args.target,
        output_dir=args.output
    )

if __name__ == '__main__':
    main() 