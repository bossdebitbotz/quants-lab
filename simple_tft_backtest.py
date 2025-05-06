#!/usr/bin/env python3

"""
Simple TFT Ensemble Backtest

This script runs a simplified backtest of the TFT ensemble model on recent orderbook data.
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
import psycopg2
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_backtest")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def get_db_connection():
    """Create a database connection"""
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

def load_orderbook_data(days=7):
    """Load orderbook data from the database"""
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query to get orderbook summary data
        query = """
        SELECT
            trading_pair,
            timestamp,
            best_bid,
            best_ask,
            mid_price,
            spread,
            spread_pct,
            bid_quantity,
            ask_quantity,
            imbalance
        FROM
            order_book_summary
        WHERE
            timestamp BETWEEN %s AND %s
        ORDER BY
            timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        logger.info(f"Loaded {len(df)} rows of orderbook data")
        return df
        
    except Exception as e:
        logger.error(f"Error loading orderbook data: {e}")
        return None

def preprocess_data(df):
    """Preprocess orderbook data for TFT model"""
    if df is None or len(df) == 0:
        return None
    
    # Make a copy of the dataframe
    df_features = df.copy()
    
    # Calculate returns at different timeframes
    df_features['returns_10sec'] = df_features['mid_price'].pct_change(1)
    df_features['returns_30sec'] = df_features['mid_price'].pct_change(3) 
    df_features['returns_1min'] = df_features['mid_price'].pct_change(6)
    
    # Calculate order book imbalance
    total_volume = df_features['bid_quantity'] + df_features['ask_quantity']
    df_features['buy_sell_imbalance'] = (df_features['bid_quantity'] - df_features['ask_quantity']) / total_volume
    
    # Calculate volatility (rolling standard deviation of returns)
    df_features['volatility_1min'] = df_features['returns_10sec'].rolling(window=6).std()
    
    # Create placeholder features 
    df_features['new_bid_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['new_ask_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['canceled_bid_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['canceled_ask_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['executed_bid_orders'] = np.random.randint(0, 5, size=len(df_features))
    df_features['executed_ask_orders'] = np.random.randint(0, 5, size=len(df_features))
    
    # Create target variables
    df_features['target'] = df_features['returns_10sec'].shift(-1) 
    
    # Create MA-adjusted target
    window_size = 30
    df_features['target_ma'] = df_features['target'].rolling(window=window_size).mean()
    df_features['target_ma_adjusted'] = df_features['target'] - df_features['target_ma']
    
    # Fill NaN values
    df_features = df_features.fillna(0)
    
    return df_features

def normalize_features(df, feature_columns):
    """Normalize features using z-score normalization"""
    df_scaled = df.copy()
    
    for col in feature_columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df_scaled[col] = (df[col] - mean) / std
        else:
            df_scaled[col] = 0
    
    return df_scaled

def simulate_random_trading(df, threshold=0.002, position_sizing=True):
    """
    Simulate trading with random predictions as a baseline
    
    Since we don't have actual TFT models to use, this function generates 
    random predictions and simulates trading based on them.
    """
    # Generate random predictions (as a baseline)
    np.random.seed(42)  # For reproducibility
    predictions = np.random.normal(0, 0.003, size=len(df))
    
    # Get actual returns
    actuals = df['returns_10sec'].shift(-1).values
    
    # Trading variables
    capital = 1.0
    position = 0  # 0=none, 1=long, -1=short
    fee = 0.001  # 0.1% fee per trade
    
    capital_history = [capital]
    position_history = [position]
    trades = []
    
    # Simulate trading
    for i in range(1, len(predictions)):
        pred = predictions[i]
        actual = actuals[i] if i < len(actuals) else 0
        
        # Determine position size
        position_size = 1.0
        if position_sizing:
            strength = min(abs(pred) / threshold, 1.0)
            position_size = max(0.1, strength)
        
        # Trading logic
        if position == 0:  # No position
            if pred > threshold:
                # Open long position
                position = 1
                capital *= (1 - fee)  # Pay fee
                trades.append({
                    'timestamp': df.index[i] if hasattr(df, 'index') else i,
                    'action': 'buy',
                    'price': df['mid_price'].iloc[i],
                    'capital': capital
                })
            elif pred < -threshold:
                # Open short position
                position = -1
                capital *= (1 - fee)  # Pay fee
                trades.append({
                    'timestamp': df.index[i] if hasattr(df, 'index') else i,
                    'action': 'sell',
                    'price': df['mid_price'].iloc[i],
                    'capital': capital
                })
        
        elif position == 1:  # Long position
            if pred < 0:
                # Close long position
                capital *= (1 + actual - fee)  # Apply return and pay fee
                position = 0
                trades.append({
                    'timestamp': df.index[i] if hasattr(df, 'index') else i,
                    'action': 'close_long',
                    'price': df['mid_price'].iloc[i],
                    'capital': capital
                })
            else:
                # Hold long position
                capital *= (1 + actual * position_size)  # Apply return
        
        elif position == -1:  # Short position
            if pred > 0:
                # Close short position
                capital *= (1 - actual - fee)  # Apply return and pay fee
                position = 0
                trades.append({
                    'timestamp': df.index[i] if hasattr(df, 'index') else i,
                    'action': 'close_short',
                    'price': df['mid_price'].iloc[i],
                    'capital': capital
                })
            else:
                # Hold short position
                capital *= (1 - actual * position_size)  # Apply return
        
        # Record position and capital
        position_history.append(position)
        capital_history.append(capital)
    
    # Calculate metrics
    if len(trades) > 0:
        returns = np.diff(capital_history) / np.array(capital_history[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(365 * 24 * 60 * 6)  # Assuming 10-second data
    else:
        returns = []
        sharpe_ratio = 0
        annualized_sharpe = 0
    
    return {
        'final_capital': capital,
        'return': capital - 1.0,
        'sharpe_ratio': annualized_sharpe,
        'total_trades': len(trades),
        'positions': position_history,
        'capital_history': capital_history,
        'trades': trades,
        'predictions': predictions,
        'actuals': actuals
    }

def plot_results(results, output_dir):
    """Plot backtest results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['capital_history'])
    plt.title(f'Equity Curve (Final: {results["final_capital"]:.4f})')
    plt.xlabel('Bar')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()
    
    # Plot predictions vs actuals (sample)
    sample_size = min(1000, len(results['predictions']))
    sample_start = len(results['predictions']) // 2 - sample_size // 2
    sample_end = sample_start + sample_size
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['actuals'][sample_start:sample_end], label='Actual Returns', alpha=0.5)
    plt.plot(results['predictions'][sample_start:sample_end], label='Predicted Returns', alpha=0.5)
    plt.title('Actual vs Predicted Returns (Sample)')
    plt.xlabel('Bar')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actuals.png'))
    plt.close()
    
    # Plot positions
    plt.figure(figsize=(12, 6))
    plt.plot(results['positions'])
    plt.title('Position History (1=Long, -1=Short, 0=No Position)')
    plt.xlabel('Bar')
    plt.ylabel('Position')
    plt.yticks([-1, 0, 1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_history.png'))
    plt.close()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple TFT Backtest')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use')
    parser.add_argument('--threshold', type=float, default=0.002, help='Trading threshold')
    parser.add_argument('--position-sizing', action='store_true', help='Enable adaptive position sizing')
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Output directory')
    args = parser.parse_args()
    
    logger.info(f"Starting backtest with {args.days} days of data")
    
    # Load orderbook data
    df = load_orderbook_data(days=args.days)
    
    if df is None or len(df) == 0:
        logger.error("Failed to load orderbook data. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(df)} rows of orderbook data")
    
    # Preprocess data
    df_features = preprocess_data(df)
    
    if df_features is None:
        logger.error("Failed to preprocess data. Exiting.")
        return 1
    
    logger.info("Data preprocessing completed")
    
    # Since we don't have actual TFT models to use, we'll simulate with random predictions
    logger.info("Running trading simulation with random predictions as baseline")
    results = simulate_random_trading(
        df=df_features,
        threshold=args.threshold,
        position_sizing=args.position_sizing
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot results
    logger.info("Plotting results")
    plot_results(results, args.output_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'backtest_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        json_results = {
            'parameters': vars(args),
            'metrics': {
                'final_capital': float(results['final_capital']),
                'return': float(results['return']),
                'sharpe_ratio': float(results['sharpe_ratio']),
                'total_trades': int(results['total_trades'])
            }
        }
        json.dump(json_results, f, indent=4)
    
    # Print summary
    logger.info("\n=== Backtest Results ===")
    logger.info(f"Initial Capital: 1.0")
    logger.info(f"Final Capital: {results['final_capital']:.4f}")
    logger.info(f"Return: {results['return']:.4f} ({results['return']*100:.2f}%)")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 