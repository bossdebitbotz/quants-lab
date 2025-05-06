#!/usr/bin/env python3

"""
TFT Ensemble Model Backtester for 7 Days of Orderbook Data

This script runs a backtest on 7 days of orderbook data using the TFT ensemble models.

Configuration:
- Directional model: models/directional_tft_20250502_134913.pt
- Downward specialist model: models/downward_specialist_tft_20250502_140300.pt
- Target Variable: target_ma_adjusted
- Ensemble Method: selective
- Ensemble Threshold: 0.002
- Trading Parameters:
  - Position Sizing: 10-50% of capital based on prediction strength
  - Stop Loss: 0.2%
  - Take Profit: 0.4%
  - Max Holding Period: 8 bars
  - Transaction Fee: 0.01%
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import psycopg2
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler

# Import our custom TFT implementation
from backtest_tft import TemporalFusionTransformer, DirectionalTFT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_orderbook_backtester")

# Default models
DIRECTIONAL_MODEL_PATH = 'models/directional_tft_20250502_134913.pt'
DOWNWARD_MODEL_PATH = 'models/downward_specialist_tft_20250502_140300.pt'

# Default database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Default trading parameters
THRESHOLD = 0.002
STOP_LOSS = 0.002
TAKE_PROFIT = 0.004
MAX_HOLDING = 8
FEE = 0.0001  # 0.01%

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def fetch_orderbook_data(days=7):
    """Fetch orderbook data from the database"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to connect to database")
            return None
            
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query for orderbook data
        query = """
        SELECT
            timestamp,
            best_bid AS bid_price,
            best_ask AS ask_price,
            (best_bid + best_ask) / 2 AS mid_price,
            (best_ask - best_bid) AS spread,
            ((best_ask - best_bid) / best_bid) AS spread_pct,
            bid_quantity,
            ask_quantity,
            (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity) AS imbalance
        FROM
            order_book_summary
        WHERE
            timestamp BETWEEN %s AND %s
        ORDER BY
            timestamp
        """
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(start_date, end_date),
            parse_dates=['timestamp']
        )
        
        logger.info(f"Retrieved {len(df)} orderbook records")
        conn.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def add_features(df):
    """Add derived features to the orderbook data"""
    df_feat = df.copy()
    
    # Calculate returns
    df_feat['returns_10sec'] = df_feat['mid_price'].pct_change()
    df_feat['returns_30sec'] = df_feat['mid_price'].pct_change(3)
    df_feat['returns_1min'] = df_feat['mid_price'].pct_change(6)
    
    # Calculate moving averages
    df_feat['price_ma_5'] = df_feat['mid_price'].rolling(5).mean()
    df_feat['price_ma_10'] = df_feat['mid_price'].rolling(10).mean()
    df_feat['price_ma_30'] = df_feat['mid_price'].rolling(30).mean()
    
    # Calculate volatility
    df_feat['volatility_10'] = df_feat['returns_10sec'].rolling(10).std()
    df_feat['volatility_30'] = df_feat['returns_10sec'].rolling(30).std()
    df_feat['volatility_1min'] = df_feat['returns_10sec'].rolling(6).std()
    
    # Calculate order book imbalance features
    df_feat['imbalance_ma_5'] = df_feat['imbalance'].rolling(5).mean()
    df_feat['imbalance_ma_10'] = df_feat['imbalance'].rolling(10).mean()
    
    # Calculate spread features
    df_feat['spread_ma_5'] = df_feat['spread'].rolling(5).mean()
    df_feat['spread_ma_10'] = df_feat['spread'].rolling(10).mean()
    
    # Calculate target (future return)
    df_feat['target'] = df_feat['returns_10sec'].shift(-1)
    
    # Calculate MA-adjusted target
    df_feat['target_ma'] = df_feat['returns_10sec'].rolling(10).mean()
    df_feat['target_ma_adjusted'] = df_feat['target'] - df_feat['target_ma'].shift(-1)
    
    # Drop NaN values
    df_feat = df_feat.dropna()
    
    return df_feat

def prepare_features_for_model(df, context_length=30):
    """
    Prepare features for TFT model.
    
    Args:
        df: DataFrame with features
        context_length: Context length for the TFT model (lookback window)
        
    Returns:
        X_tensor: Input tensor for the model [batch_size, seq_len, n_features]
        y_array: Target values as numpy array
        timestamps: List of timestamps for each prediction
    """
    # List of features to use for prediction
    feature_cols = [
        'returns_10sec', 'returns_30sec', 'returns_1min', 
        'price_ma_5', 'price_ma_10', 'price_ma_30',
        'volatility_10', 'volatility_30', 'volatility_1min',
        'imbalance', 'imbalance_ma_5', 'imbalance_ma_10',
        'spread', 'spread_pct', 'spread_ma_5', 'spread_ma_10'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )
    
    # Add back the timestamp column
    scaled_features['timestamp'] = df['timestamp']
    
    # Target variable
    target_col = 'target_ma_adjusted'
    y_array = df[target_col].values
    
    # Create sliding windows for the context
    X_windows = []
    timestamps = []
    
    for i in range(context_length, len(scaled_features)):
        window = scaled_features.iloc[i-context_length:i][feature_cols].values
        X_windows.append(window)
        timestamps.append(scaled_features.iloc[i]['timestamp'])
    
    # Convert to tensor
    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
    
    # Adjust y_array to match the length after windowing
    y_array = y_array[context_length:]
    
    logger.info(f"Prepared {len(X_windows)} feature windows with shape {X_tensor.shape}")
    
    return X_tensor, y_array, timestamps

def load_model(model_path):
    """
    Load a PyTorch model from the given path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        The loaded model or None if loading fails
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Add safe globals for loading Lightning models
        import torch.serialization
        try:
            # Allow necessary globals for Lightning models
            torch.serialization.add_safe_globals(['lightning_fabric.utilities.data.AttributeDict'])
        except:
            logger.warning("Could not add safe globals for Lightning, attempting load without weights_only")
        
        # Try loading with weights_only=False (for trusted files)
        try:
            model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        logger.info(f"Model data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
        
        # Create appropriate model based on filename
        is_directional = 'directional' in model_path.lower()
        is_downward = 'downward' in model_path.lower()
        
        # Get config from model data
        if isinstance(model_data, dict) and 'config' in model_data:
            config = model_data['config']
            logger.info(f"Using config from model: {config}")
        else:
            # Default config
            config = {
                'hidden_size': 128,
                'lstm_layers': 2,
                'num_attention_heads': 4,
                'dropout': 0.2,
                'context_length': 20,
                'prediction_length': 1,
                'bias_correction': True
            }
            logger.info(f"Using default config: {config}")
        
        # Update config with feature count
        config['time_varying_real_variables'] = 16  # Number of features we're using
        
        # Create appropriate model type
        if is_directional or is_downward:
            # Add directional-specific parameters
            config['loss_fn'] = 'directional'
            config['alpha'] = 0.9  # High weight for directional accuracy
            
            # Create model
            model = DirectionalTFT(**config)
            logger.info(f"Created DirectionalTFT model")
        else:
            # Standard TFT
            model = TemporalFusionTransformer(**config)
            logger.info(f"Created standard TFT model")
        
        # Load state dict if available
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            model.load_state_dict(model_data['model_state_dict'])
            logger.info(f"Loaded model state dict")
        
        # Set to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_ensemble_predictions(X_tensor, directional_model, downward_model, method="selective"):
    """
    Generate predictions using the ensemble of TFT models.
    
    Args:
        X_tensor: Input tensor [batch_size, seq_len, n_features]
        directional_model: Main directional model
        downward_model: Specialist model for downward movements
        method: Ensemble method ("average", "weighted", "selective")
        
    Returns:
        Numpy array of ensemble predictions
    """
    # Run predictions with both models
    with torch.no_grad():
        directional_output = directional_model(X_tensor)
        downward_output = downward_model(X_tensor)
        
        # Extract predictions from model outputs
        if isinstance(directional_output, dict):
            directional_preds = directional_output['prediction'].cpu().numpy()
        else:
            directional_preds = directional_output.cpu().numpy()
            
        if isinstance(downward_output, dict):
            downward_preds = downward_output['prediction'].cpu().numpy()
        else:
            downward_preds = downward_output.cpu().numpy()
    
    # Reshape if needed
    if len(directional_preds.shape) > 1:
        directional_preds = directional_preds.flatten()
    if len(downward_preds.shape) > 1:
        downward_preds = downward_preds.flatten()
    
    logger.info(f"Generated predictions - directional: shape={directional_preds.shape}, mean={np.mean(directional_preds):.6f}, std={np.std(directional_preds):.6f}")
    logger.info(f"Generated predictions - downward: shape={downward_preds.shape}, mean={np.mean(downward_preds):.6f}, std={np.std(downward_preds):.6f}")
    
    # Generate ensemble predictions based on method
    if method == "average":
        # Simple average of both models
        ensemble_preds = (directional_preds + downward_preds) / 2
    
    elif method == "weighted":
        # Example: weight the specialist model higher for negative predictions
        weights = np.where(directional_preds < 0, 
                          [0.3, 0.7],   # More weight to downward model for negative predictions
                          [0.7, 0.3])   # More weight to directional model for positive predictions
        
        ensemble_preds = weights[:, 0] * directional_preds + weights[:, 1] * downward_preds
    
    elif method == "selective":
        # Use directional model for upward predictions, downward model for negative
        ensemble_preds = np.where(directional_preds >= 0,
                                 directional_preds,    # Use directional for positive
                                 downward_preds)       # Use downward for negative
    
    else:
        # Default to directional model only
        ensemble_preds = directional_preds
    
    logger.info(f"Ensemble predictions: shape={ensemble_preds.shape}, mean={np.mean(ensemble_preds):.6f}, std={np.std(ensemble_preds):.6f}")
    
    return ensemble_preds

def simulate_trading(timestamps, predictions, actuals, threshold=THRESHOLD, 
                    stop_loss=STOP_LOSS, take_profit=TAKE_PROFIT, 
                    max_holding_period=MAX_HOLDING, transaction_fee=FEE,
                    use_position_sizing=True, use_dynamic_threshold=True):
    """
    Simulate trading based on ensemble predictions
    
    Args:
        timestamps: List of timestamps for each prediction
        predictions: Numpy array of ensemble predictions
        actuals: Numpy array of actual returns
        threshold: Threshold for entering positions
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        max_holding_period: Maximum holding period in bars
        transaction_fee: Transaction fee per trade
        use_position_sizing: Whether to use adaptive position sizing
        use_dynamic_threshold: Whether to adjust thresholds based on volatility
        
    Returns:
        Dictionary with trading results and DataFrame with trade details
    """
    logger.info(f"Running trading simulation with threshold={threshold}, SL={stop_loss}, TP={take_profit}")
    
    # Create a DataFrame with timestamps, predictions, and actuals
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions,
        'actual': actuals
    })
    
    # Add volatility for dynamic thresholds
    if use_dynamic_threshold:
        # Calculate rolling volatility (std of actual returns)
        df['volatility'] = df['actual'].rolling(window=20).std()
        
        # Fill NaN with the mean
        mean_vol = df['volatility'].mean()
        df['volatility'] = df['volatility'].fillna(mean_vol)
        
        # Adjust threshold based on volatility
        # Higher volatility = higher threshold
        df['adjusted_threshold'] = threshold * (df['volatility'] / mean_vol)
    else:
        df['adjusted_threshold'] = threshold
    
    # Initialize variables
    positions = np.zeros(len(df))  # 0=no position, 1=long, -1=short
    position_sizes = np.zeros(len(df))
    entry_indices = np.zeros(len(df), dtype=int) - 1
    holding_periods = np.zeros(len(df), dtype=int)
    capital = np.ones(len(df))
    trades = []
    
    # Simulation loop
    for i in range(len(df)):
        # Get current prediction and threshold
        pred = df.iloc[i]['prediction']
        current_threshold = df.iloc[i]['adjusted_threshold']
        
        # For first bar, initialize capital
        if i == 0:
            continue
            
        # Update holding period for existing positions
        if positions[i-1] != 0:
            positions[i] = positions[i-1]
            position_sizes[i] = position_sizes[i-1]
            holding_periods[i] = holding_periods[i-1] + 1
            entry_indices[i] = entry_indices[i-1]
            
            # Check exit conditions
            entry_idx = entry_indices[i]
            if entry_idx >= 0:
                # Calculate cumulative P&L since entry
                cum_returns = np.sum(df.iloc[entry_idx+1:i+1]['actual']) if entry_idx < i else 0
                position_pnl = positions[i] * cum_returns
                
                # Exit logic
                exit_position = False
                exit_reason = ""
                
                # Stop loss
                if position_pnl < -stop_loss:
                    exit_position = True
                    exit_reason = "stop_loss"
                
                # Take profit
                elif position_pnl > take_profit:
                    exit_position = True
                    exit_reason = "take_profit"
                
                # Max holding period
                elif holding_periods[i] >= max_holding_period:
                    exit_position = True
                    exit_reason = "max_holding"
                
                # Exit position if needed
                if exit_position:
                    # Record trade
                    trades.append({
                        'entry_time': df.iloc[entry_idx]['timestamp'],
                        'exit_time': df.iloc[i]['timestamp'],
                        'direction': 'long' if positions[i] > 0 else 'short',
                        'size': position_sizes[i],
                        'entry_price': None,  # We don't track actual prices
                        'exit_price': None,   # We don't track actual prices
                        'pnl': position_pnl,
                        'pnl_after_fees': position_pnl - transaction_fee * 2,  # Entry + exit fees
                        'holding_period': holding_periods[i],
                        'exit_reason': exit_reason
                    })
                    
                    # Update capital (account for fees)
                    capital[i] = capital[i-1] * (1 + position_sizes[i] * position_pnl - 2 * transaction_fee * position_sizes[i])
                    
                    # Reset position
                    positions[i] = 0
                    position_sizes[i] = 0
                    holding_periods[i] = 0
                    entry_indices[i] = -1
                else:
                    # If still holding, update capital
                    capital[i] = capital[i-1] * (1 + position_sizes[i] * df.iloc[i]['actual'])
            
            # Skip the rest of the loop if already in a position
            continue
        
        # Entry logic (when not in a position)
        # Scale copy of previous capital if no position
        capital[i] = capital[i-1]
        
        # Check if prediction exceeds threshold for entry
        if abs(pred) > current_threshold:
            # Determine position direction
            direction = 1 if pred > 0 else -1
            
            # Determine position size if using adaptive sizing
            if use_position_sizing:
                # Scale position size with prediction strength (10% to 50%)
                min_size = 0.1
                max_size = 0.5
                # Normalize prediction to [0, 1] range for sizing
                pred_strength = min(abs(pred) / (current_threshold * 5), 1.0)
                position_size = min_size + (max_size - min_size) * pred_strength
            else:
                position_size = 0.1  # Fixed 10% position size
            
            # Enter position
            positions[i] = direction
            position_sizes[i] = position_size
            entry_indices[i] = i
            holding_periods[i] = 0
        
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate trading metrics
    if len(trades) > 0:
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['pnl_after_fees'] > 0)
        win_rate = winning_trades / total_trades
        
        avg_win = np.mean([trade['pnl_after_fees'] for trade in trades if trade['pnl_after_fees'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade['pnl_after_fees'] for trade in trades if trade['pnl_after_fees'] <= 0]) if total_trades - winning_trades > 0 else 0
        
        total_return = capital[-1] - 1.0
        sharpe_ratio = (total_return / (np.std(capital[1:] / capital[:-1] - 1) * np.sqrt(len(capital)))) if np.std(capital[1:] / capital[:-1] - 1) > 0 else 0
        
        avg_holding = np.mean([trade['holding_period'] for trade in trades])
        
        max_drawdown = 0
        peak = 1.0
        for cap in capital:
            if cap > peak:
                peak = cap
            drawdown = (peak - cap) / peak
            max_drawdown = max(max_drawdown, drawdown)
    else:
        total_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        total_return = 0
        sharpe_ratio = 0
        avg_holding = 0
        max_drawdown = 0
    
    # Results dictionary
    results = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'avg_holding_period': avg_holding,
        'max_drawdown': max_drawdown,
        'final_equity': capital[-1],
        'capital_curve': capital,
        'timestamps': timestamps,
        'prediction_mean': np.mean(predictions),
        'prediction_std': np.std(predictions)
    }
    
    return results, trades_df

def plot_results(results, df, output_dir="backtest_results"):
    """
    Plot the backtest results.
    
    Args:
        results: Dictionary with backtest results
        df: DataFrame with trade data
        output_dir: Output directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure for equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['timestamps'], results['capital_curve'], 'b-')
    plt.title(f'Equity Curve (Total Return: {results["total_return"]:.2%})')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=300)
    plt.close()
    
    # Plot trade distribution
    if len(df) > 0:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['pnl_after_fees'], bins=20, kde=True)
        plt.title(f'Trade P&L Distribution (Win Rate: {results["win_rate"]:.2%})')
        plt.xlabel('P&L')
        plt.ylabel('Count')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'), dpi=300)
        plt.close()
        
        # Plot holding periods
        plt.figure(figsize=(12, 6))
        sns.histplot(df['holding_period'], bins=range(1, max(df['holding_period'].astype(int)) + 2), kde=False)
        plt.title(f'Holding Period Distribution (Avg: {results["avg_holding_period"]:.2f} bars)')
        plt.xlabel('Holding Period (bars)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'holding_periods.png'), dpi=300)
        plt.close()
        
    # Save results to JSON
    with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
        # Convert timestamps to strings for JSON serialization
        results_json = results.copy()
        results_json['timestamps'] = [str(ts) for ts in results_json['timestamps']]
        # Convert numpy arrays to lists
        results_json['capital_curve'] = results_json['capital_curve'].tolist()
        
        json.dump(results_json, f, indent=4)
    
    # Save trades to CSV
    if len(df) > 0:
        df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

def main():
    """
    Main function to run the TFT ensemble backtester.
    """
    parser = argparse.ArgumentParser(description='TFT Ensemble Model Backtester')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use (default: 7)')
    parser.add_argument('--directional-model', type=str, default=DIRECTIONAL_MODEL_PATH, help='Path to directional model')
    parser.add_argument('--downward-model', type=str, default=DOWNWARD_MODEL_PATH, help='Path to downward specialist model')
    parser.add_argument('--ensemble-method', type=str, default='selective', choices=['average', 'weighted', 'selective'], help='Ensemble method')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Base threshold for trade entry')
    parser.add_argument('--stop-loss', type=float, default=STOP_LOSS, help='Stop loss percentage') 
    parser.add_argument('--take-profit', type=float, default=TAKE_PROFIT, help='Take profit percentage')
    parser.add_argument('--max-holding', type=int, default=MAX_HOLDING, help='Maximum holding period')
    parser.add_argument('--fee', type=float, default=FEE, help='Transaction fee per trade')
    parser.add_argument('--target-col', type=str, default='target_ma_adjusted', help='Target column')
    parser.add_argument('--context-length', type=int, default=30, help='Context length for TFT model')
    parser.add_argument('--position-sizing', action='store_true', default=True, help='Enable adaptive position sizing')
    parser.add_argument('--no-position-sizing', action='store_false', dest='position_sizing', help='Disable adaptive position sizing')
    parser.add_argument('--dynamic-threshold', action='store_true', default=True, help='Enable dynamic thresholds')
    parser.add_argument('--no-dynamic-threshold', action='store_false', dest='dynamic_threshold', help='Disable dynamic thresholds')
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Output directory for results')
    args = parser.parse_args()
    
    # Print configuration
    logger.info("=== TFT Ensemble Model Backtester ===")
    logger.info(f"Directional Model: {args.directional_model}")
    logger.info(f"Downward Model: {args.downward_model}")
    logger.info(f"Ensemble Method: {args.ensemble_method}")
    logger.info(f"Data Period: {args.days} days")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Stop Loss: {args.stop_loss}")
    logger.info(f"Take Profit: {args.take_profit}")
    logger.info(f"Max Holding Period: {args.max_holding}")
    logger.info(f"Transaction Fee: {args.fee}")
    logger.info(f"Target Column: {args.target_col}")
    logger.info(f"Position Sizing: {'Enabled' if args.position_sizing else 'Disabled'}")
    logger.info(f"Dynamic Threshold: {'Enabled' if args.dynamic_threshold else 'Disabled'}")
    
    # Step 1: Load order book data from the database
    logger.info(f"Loading {args.days} days of orderbook data from database")
    df = fetch_orderbook_data(days=args.days)
    if df is None or len(df) < 100:
        logger.error("Insufficient data to perform backtest")
        return
    
    # Step 2: Add derived features and target variables
    logger.info("Adding derived features")
    df_features = add_features(df)
    
    # Step 3: Prepare data for TFT model
    logger.info(f"Preparing features with context length {args.context_length}")
    X_tensor, y_array, timestamps = prepare_features_for_model(
        df_features, 
        context_length=args.context_length
    )
    
    # Step 4: Load models
    logger.info("Loading TFT models")
    directional_model = load_model(args.directional_model)
    downward_model = load_model(args.downward_model)
    
    if directional_model is None or downward_model is None:
        logger.error("Failed to load models")
        return
    
    # Step 5: Generate ensemble predictions
    logger.info(f"Generating ensemble predictions using {args.ensemble_method} method")
    ensemble_preds = generate_ensemble_predictions(
        X_tensor,
        directional_model,
        downward_model,
        method=args.ensemble_method
    )
    
    # Step 6: Run trading simulation
    logger.info("Running trading simulation")
    trading_results, trades_df = simulate_trading(
        timestamps=timestamps,
        predictions=ensemble_preds,
        actuals=y_array,
        threshold=args.threshold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_period=args.max_holding,
        transaction_fee=args.fee,
        use_position_sizing=args.position_sizing,
        use_dynamic_threshold=args.dynamic_threshold
    )
    
    # Step 7: Plot and save results
    logger.info(f"Plotting and saving results to {args.output_dir}")
    plot_results(trading_results, trades_df, output_dir=args.output_dir)
    
    # Step 8: Display summary
    logger.info("\n=== Backtest Results ===")
    logger.info(f"Total Trades: {trading_results['total_trades']}")
    logger.info(f"Win Rate: {trading_results['win_rate']:.2%}")
    logger.info(f"Total Return: {trading_results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
    logger.info(f"Average Holding Period: {trading_results['avg_holding_period']:.2f} bars")
    logger.info(f"Max Drawdown: {trading_results['max_drawdown']:.2%}")
    logger.info(f"Final Equity: {trading_results['final_equity']:.4f}")
    logger.info(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 