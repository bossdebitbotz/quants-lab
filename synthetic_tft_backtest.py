#!/usr/bin/env python3

"""
Synthetic TFT Backtest

This script performs a synthetic backtest of the TFT ensemble trading strategy
by simulating model predictions based on known performance characteristics
rather than using the actual models.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import psycopg2
import random
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("synthetic_tft_backtest")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Trading strategy parameters
POSITION_SIZING = True
MIN_POSITION_SIZE = 0.1
MAX_POSITION_SIZE = 0.5
POSITION_SCALING_FACTOR = 5
STOP_LOSS = 0.002  # 0.2%
TAKE_PROFIT = 0.004  # 0.4%
MAX_HOLDING_PERIOD = 8
DYNAMIC_THRESHOLDS = True
BASE_THRESHOLD = 0.002
TRANSACTION_FEE = 0.0001  # 0.01%

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

def load_recent_data(days=7):
    """
    Load recent orderbook data from the database
    
    Args:
        days: Number of days of data to load
        
    Returns:
        DataFrame with the data
    """
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to connect to database")
        return None
    
    try:
        # Calculate the start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        # Query to get orderbook summary data
        query_orderbook = """
        SELECT 
            timestamp,
            best_bid,
            best_ask,
            mid_price,
            spread,
            spread_pct,
            bid_quantity,
            ask_quantity,
            imbalance
        FROM order_book_summary
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
        """
        
        # Query to get trades data if available
        query_trades = """
        SELECT 
            timestamp,
            price,
            quantity,
            side
        FROM trades
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
        """
        
        # Load orderbook data
        df_orderbook = pd.read_sql_query(query_orderbook, conn, params=(start_date, end_date))
        
        # Load trades data
        df_trades = pd.read_sql_query(query_trades, conn, params=(start_date, end_date))
        
        logger.info(f"Retrieved {len(df_orderbook)} orderbook records and {len(df_trades)} trade records")
        
        # Check if we have enough data
        if len(df_orderbook) < 100:
            logger.warning("Not enough data for backtesting")
            return None
        
        # Check time intervals
        if len(df_orderbook) > 1:
            time_diffs = np.diff(df_orderbook['timestamp'].astype(np.int64)) / 1e9  # Convert to seconds
            median_diff = np.median(time_diffs)
            logger.info(f"Median time difference between records: {median_diff} seconds")
            
            # If data isn't 10-second bars, resample it
            if median_diff > 15:  # More than 15 seconds between records
                logger.info(f"Resampling data to 10-second intervals")
                df_orderbook = df_orderbook.set_index('timestamp').resample('10S').ffill().reset_index()
        
        # Close connection
        conn.close()
        
        return df_orderbook
    
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        if conn:
            conn.close()
        return None

def add_derived_features(df):
    """
    Add derived features to the dataframe for modeling
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        DataFrame with added features
    """
    # Make a copy to avoid modifying the original
    df_with_features = df.copy()
    
    # Add price returns for different timeframes
    df_with_features['returns_10sec'] = df_with_features['mid_price'].pct_change()
    df_with_features['returns_30sec'] = df_with_features['mid_price'].pct_change(3)
    df_with_features['returns_1min'] = df_with_features['mid_price'].pct_change(6)
    
    # Add rolling volatility
    df_with_features['volatility_1min'] = df_with_features['returns_10sec'].rolling(6).std()
    
    # Add moving average features
    df_with_features['ma_fast'] = df_with_features['mid_price'].rolling(3).mean()
    df_with_features['ma_slow'] = df_with_features['mid_price'].rolling(30).mean()
    df_with_features['ma_diff'] = df_with_features['ma_fast'] - df_with_features['ma_slow']
    
    # Create a target variable (future returns)
    df_with_features['target'] = df_with_features['returns_10sec'].shift(-1)
    
    # Create a target adjusted for moving average (better for prediction)
    df_with_features['target_ma_adjusted'] = df_with_features['target'] - df_with_features['ma_diff']
    
    # Add volume features
    df_with_features['volume_ratio'] = df_with_features['bid_quantity'] / df_with_features['ask_quantity']
    df_with_features['log_volume_ratio'] = np.log1p(df_with_features['volume_ratio'])
    
    # Add change in imbalance
    df_with_features['imbalance_change'] = df_with_features['imbalance'].diff()
    
    # Add spread change
    df_with_features['spread_change'] = df_with_features['spread'].diff()
    
    # Add bid-ask pressure
    bid_pressure = df_with_features['bid_quantity'].diff()
    ask_pressure = df_with_features['ask_quantity'].diff()
    df_with_features['ba_pressure'] = bid_pressure - ask_pressure
    
    # Fill NaN values created by diff and rolling operations
    numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
    df_with_features[numeric_cols] = df_with_features[numeric_cols].fillna(0)
    
    logger.info(f"Added derived features, final shape: {df_with_features.shape}")
    
    return df_with_features

def generate_synthetic_predictions(df, context_length=30, win_rate=0.532, correlation=0.3):
    """
    Generate synthetic predictions that mimic the expected characteristics
    of our TFT ensemble model
    
    Args:
        df: DataFrame with features
        context_length: Context window length (to skip initial rows)
        win_rate: Desired win rate for trading signals
        correlation: Correlation between predictions and actual returns
        
    Returns:
        Array of synthetic predictions
    """
    # Skip the first context_length rows as they would be used for context
    actuals = df['target_ma_adjusted'].values[context_length:]
    n_samples = len(actuals)
    
    # Create synthetic predictions that closely match future returns
    # This approach directly models the reported high win rate
    
    # Copy actuals as the base for our predictions (perfect foresight)
    perfect_preds = actuals.copy() 
    
    # Add noise to perfect foresight to get the desired win rate
    # Higher win_rate means less noise
    noise_level = 1.0 - win_rate
    noise = np.random.normal(0, np.std(actuals) * noise_level * 2, n_samples)
    
    # Mix perfect prediction with noise
    synthetic_preds = perfect_preds + noise
    
    # Scale predictions to match expected magnitude
    # The scale should align with the BASE_THRESHOLD value
    scale_factor = BASE_THRESHOLD * 3 / np.std(synthetic_preds)
    synthetic_preds = synthetic_preds * scale_factor
    
    # Verify we're close to the target win rate
    pred_signs = np.sign(synthetic_preds)
    actual_signs = np.sign(actuals)
    actual_win_rate = np.mean(pred_signs == actual_signs)
    logger.info(f"Synthetic model actual win rate: {actual_win_rate:.2%} (target: {win_rate:.2%})")
    
    # Force adjustment if we're far off
    if abs(actual_win_rate - win_rate) > 0.05:
        logger.info("Adjusting synthetic predictions to match target win rate")
        
        # Determine how many predictions to flip
        correct_preds = pred_signs == actual_signs
        incorrect_preds = ~correct_preds
        
        if actual_win_rate < win_rate:
            # Need to increase win rate
            # Flip some incorrect predictions to correct
            flip_count = int((win_rate - actual_win_rate) * n_samples)
            flip_indices = np.where(incorrect_preds)[0]
            if len(flip_indices) > 0:
                flip_indices = np.random.choice(flip_indices, min(flip_count, len(flip_indices)), replace=False)
                synthetic_preds[flip_indices] = actual_signs[flip_indices] * abs(synthetic_preds[flip_indices])
        else:
            # Need to decrease win rate
            # Flip some correct predictions to incorrect
            flip_count = int((actual_win_rate - win_rate) * n_samples)
            flip_indices = np.where(correct_preds)[0]
            if len(flip_indices) > 0:
                flip_indices = np.random.choice(flip_indices, min(flip_count, len(flip_indices)), replace=False)
                synthetic_preds[flip_indices] = -actual_signs[flip_indices] * abs(synthetic_preds[flip_indices])
    
    # Create separate predictions for directional and downward models
    directional_preds = synthetic_preds.copy()
    downward_preds = synthetic_preds.copy()
    
    # Adjust directional model to be better at upward movements
    directional_mask = actuals > 0
    directional_preds[directional_mask] *= 1.5  # Amplify correct upward predictions
    
    # Adjust downward model to be better at downward movements
    downward_mask = actuals < 0
    downward_preds[downward_mask] *= 1.5  # Amplify correct downward predictions
    
    return directional_preds, downward_preds

def simulate_trading(
    df, directional_predictions, downward_predictions, 
    threshold=BASE_THRESHOLD, 
    position_sizing=POSITION_SIZING,
    stop_loss=STOP_LOSS, 
    take_profit=TAKE_PROFIT, 
    max_holding_period=MAX_HOLDING_PERIOD,
    use_dynamic_threshold=DYNAMIC_THRESHOLDS,
    fee_per_trade=TRANSACTION_FEE
):
    """
    Simulate trading based on ensemble predictions
    
    Args:
        df: DataFrame with price data
        directional_predictions: Predictions from directional model
        downward_predictions: Predictions from downward specialist model
        threshold: Base threshold for trade entry
        position_sizing: Whether to use adaptive position sizing
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        max_holding_period: Maximum bars to hold a position
        use_dynamic_threshold: Whether to use dynamic thresholds
        fee_per_trade: Fee per trade (one way)
        
    Returns:
        Dictionary with trading metrics
    """
    # Extract targets (actual future returns)
    targets = df['target_ma_adjusted'].values[30:]  # Skip first 30 rows to align with predictions
    
    # Flatten predictions
    directional_preds = directional_predictions.flatten()
    downward_preds = downward_predictions.flatten()
    
    # Combine predictions using selective ensemble method
    predictions = np.zeros_like(directional_preds)
    
    # Use directional model for positive predictions and downward model for negative
    for i in range(len(predictions)):
        if directional_preds[i] > 0:
            predictions[i] = directional_preds[i]
        else:
            predictions[i] = downward_preds[i]
    
    # Calculate dynamic threshold if enabled
    if use_dynamic_threshold:
        # Use rolling standard deviation of targets
        rolling_window = min(30, len(targets) // 10)
        rolling_std = np.zeros_like(targets)
        
        for i in range(len(targets)):
            start_idx = max(0, i - rolling_window)
            if i > 5:
                rolling_std[i] = np.std(targets[start_idx:i+1]) 
            else:
                if i+1 > 0:
                    rolling_std[i] = np.std(targets[:i+1])
                else:
                    rolling_std[i] = 0.0
        
        # Scale threshold by volatility (higher volatility = higher threshold)
        mean_std = np.mean(rolling_std) if np.mean(rolling_std) > 0 else 1.0
        dynamic_threshold = threshold * (1 + rolling_std / mean_std)
        entry_threshold = dynamic_threshold
    else:
        entry_threshold = np.ones_like(predictions) * threshold
    
    # Initialize trading variables
    positions = np.zeros_like(predictions)  # 0: no position, 1: long, -1: short
    position_sizes = np.zeros_like(predictions)  # Fraction of capital
    entry_indices = np.zeros_like(predictions, dtype=int) - 1  # -1 means no entry
    holding_periods = np.zeros_like(predictions, dtype=int)
    
    # Capital tracking
    initial_capital = 1.0
    capital = np.ones_like(predictions) * initial_capital
    
    # Trade tracking
    trade_returns = []
    trade_durations = []
    trade_directions = []
    
    # Timestamps for trades
    entry_timestamps = []
    exit_timestamps = []
    
    # Truncate df to match prediction length
    results_df = df.iloc[30:30+len(predictions)].reset_index(drop=True)
    
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
                current_pnl = positions[i-1] * targets[i]
                
                # Exit conditions
                exit_position = False
                exit_reason = ""
                
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
                    trade_pnl = positions[i-1] * sum(targets[idx] for idx in idx_since_entry if idx < len(targets))
                    
                    # Record trade
                    trade_returns.append(trade_pnl)
                    trade_durations.append(holding_periods[i])
                    trade_directions.append(positions[i-1])
                    
                    if len(results_df) > entry_indices[i-1] and len(results_df) > i:
                        entry_timestamps.append(results_df.iloc[entry_indices[i-1]]['timestamp'])
                        exit_timestamps.append(results_df.iloc[i]['timestamp'])
                    
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
                    strength = min(1.0, predictions[i] / (entry_threshold[i] * POSITION_SCALING_FACTOR))
                    position_sizes[i] = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * strength
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
                    strength = min(1.0, abs(predictions[i]) / (entry_threshold[i] * POSITION_SCALING_FACTOR))
                    position_sizes[i] = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * strength
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
    
    # Close any remaining open position
    for i in range(len(predictions)):
        if positions[i] != 0 and i == len(predictions) - 1 and entry_indices[i] >= 0:
            # Calculate full trade return from entry to last bar
            idx_since_entry = range(entry_indices[i] + 1, i + 1)
            trade_pnl = positions[i] * sum(targets[idx] for idx in idx_since_entry if idx < len(targets))
            
            trade_returns.append(trade_pnl)
            trade_durations.append(holding_periods[i])
            trade_directions.append(positions[i])
            
            if len(results_df) > entry_indices[i] and len(results_df) > i:
                entry_timestamps.append(results_df.iloc[entry_indices[i]]['timestamp'])
                exit_timestamps.append(results_df.iloc[i]['timestamp'])
    
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
    
    # Create trades DataFrame if we have trades
    if total_trades > 0 and len(entry_timestamps) == len(exit_timestamps) == len(trade_returns):
        trades_df = pd.DataFrame({
            'entry_time': entry_timestamps,
            'exit_time': exit_timestamps,
            'direction': trade_directions,
            'duration': trade_durations,
            'return': trade_returns
        })
    else:
        trades_df = None
    
    # Return metrics and results
    metrics = {
        'threshold': threshold,
        'total_trades': int(total_trades),
        'win_rate': float(win_rate) if total_trades > 0 else 0,
        'total_return': float(total_return),
        'avg_return_per_trade': float(avg_return_per_trade),
        'sharpe_ratio': float(sharpe),
        'avg_trade_duration': float(np.mean(trade_durations)) if trade_durations else 0,
        'final_equity': float(final_equity),
        'capital_curve': capital.tolist(),
        'trades_df': trades_df
    }
    
    return metrics

def plot_results(metrics, output_dir='plots'):
    """
    Generate and save performance plots.
    
    Args:
        metrics: Trading metrics dictionary
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['capital_curve'])
    plt.title(f'Equity Curve (Threshold={metrics["threshold"]:.4f})')
    plt.xlabel('Bar #')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()
    
    # Only create trade distribution plots if we have enough trades
    if metrics['total_trades'] >= 10:
        # Plot trade returns histogram
        plt.figure(figsize=(12, 6))
        if metrics['trades_df'] is not None:
            sns.histplot(metrics['trades_df']['return'], bins=min(50, metrics['total_trades']//2), kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f'Trade Returns Distribution (Win Rate: {metrics["win_rate"]:.2%})')
            plt.xlabel('Return')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'trade_returns.png'))
            plt.close()
            
            # Plot trade duration histogram
            plt.figure(figsize=(12, 6))
            sns.histplot(metrics['trades_df']['duration'], bins=min(20, metrics['total_trades']//5), kde=True)
            plt.title(f'Trade Duration Distribution (Avg: {metrics["avg_trade_duration"]:.2f} bars)')
            plt.xlabel('Duration (bars)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'trade_durations.png'))
            plt.close()
            
            # Plot cumulative returns over time
            plt.figure(figsize=(12, 6))
            metrics['trades_df']['cumulative_return'] = metrics['trades_df']['return'].cumsum()
            plt.plot(metrics['trades_df'].index, metrics['trades_df']['cumulative_return'])
            plt.title('Cumulative Trade Returns')
            plt.xlabel('Trade #')
            plt.ylabel('Cumulative Return')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
            plt.close()

def main():
    """
    Main function to run the backtest.
    """
    parser = argparse.ArgumentParser(description='Synthetic TFT Ensemble Backtest')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use')
    parser.add_argument('--threshold', type=float, default=0.002, 
                        help='Base threshold for trading')
    parser.add_argument('--output-dir', type=str, default='synthetic_backtest_results',
                        help='Directory to save results and plots')
    parser.add_argument('--win-rate', type=float, default=0.532,
                        help='Target win rate for synthetic model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store threshold value
    threshold = args.threshold
    
    # Load data
    df = load_recent_data(days=args.days)
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Add derived features
    df = add_derived_features(df)
    
    # Generate synthetic predictions
    directional_predictions, downward_predictions = generate_synthetic_predictions(
        df, 
        context_length=30,
        win_rate=args.win_rate,
        correlation=0.3
    )
    
    # Simulate trading
    logger.info("Simulating trading...")
    metrics = simulate_trading(
        df, 
        directional_predictions, 
        downward_predictions,
        threshold=threshold
    )
    
    # Plot and save results
    logger.info("Plotting results...")
    plot_results(metrics, output_dir=args.output_dir)
    
    # Print summary
    logger.info("\n=== Trading Summary ===")
    logger.info(f"Base Threshold: {threshold}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Average Trade Duration: {metrics['avg_trade_duration']:.2f} bars")
    logger.info(f"Average Return per Trade: {metrics['avg_return_per_trade']:.4f}")
    
    # Save metrics to JSON file
    metrics_json = {k: v for k, v in metrics.items() if k != 'trades_df' and k != 'capital_curve'}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save summary to text file
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("=== Synthetic TFT Ensemble Backtest ===\n\n")
        f.write(f"Data Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Base Threshold: {threshold}\n")
        f.write(f"Synthetic Model Win Rate Target: {args.win_rate:.2%}\n\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"Actual Win Rate: {metrics['win_rate']:.2%}\n")
        f.write(f"Total Return: {metrics['total_return']:.2%}\n")
        f.write(f"Final Equity: {metrics['final_equity']:.4f}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"Average Trade Duration: {metrics['avg_trade_duration']:.2f} bars\n")
        f.write(f"Average Return per Trade: {metrics['avg_return_per_trade']:.4f}\n\n")
        f.write(f"Trading Parameters:\n")
        f.write(f"- Stop Loss: {STOP_LOSS*100:.2f}%\n")
        f.write(f"- Take Profit: {TAKE_PROFIT*100:.2f}%\n")
        f.write(f"- Max Holding Period: {MAX_HOLDING_PERIOD} bars\n")
        f.write(f"- Position Sizing: {'Enabled' if POSITION_SIZING else 'Disabled'}\n")
        f.write(f"- Transaction Fee: {TRANSACTION_FEE*100:.4f}% per trade\n")
    
    # Save trades to CSV if available
    if metrics['trades_df'] is not None:
        metrics['trades_df'].to_csv(os.path.join(args.output_dir, 'trades.csv'), index=False)
    
    logger.info(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 