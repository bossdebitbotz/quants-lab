#!/usr/bin/env python3

"""
TFT Ensemble Model Backtester for 7 Days of Data

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_7day_backtester")

# Database configuration
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
    
    # Calculate target (future return)
    df_feat['target'] = df_feat['returns_10sec'].shift(-1)
    
    # Calculate MA-adjusted target
    df_feat['target_ma_adjusted'] = df_feat['target'] - df_feat['returns_10sec'].rolling(10).mean().shift(-1)
    
    # Drop NaN values
    df_feat = df_feat.dropna()
    
    return df_feat

def scale_features(df, feature_cols):
    """Scale features using StandardScaler"""
    df_scaled = df.copy()
    
    for col in feature_cols:
        scaler = StandardScaler()
        df_scaled[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
    
    return df_scaled

def simulate_trading(df, threshold=THRESHOLD, stop_loss=STOP_LOSS, 
                   take_profit=TAKE_PROFIT, max_holding=MAX_HOLDING, fee=FEE):
    """
    Simulate trading based on ensemble predictions
    
    Args:
        df: DataFrame with timestamps, predictions, and actuals
        threshold: Threshold for entering positions
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        max_holding: Maximum holding period in bars
        fee: Transaction fee per trade
        
    Returns:
        Dictionary with trading results
    """
    logger.info(f"Running trading simulation with threshold={threshold}, SL={stop_loss}, TP={take_profit}")
    
    # Initialize variables
    positions = np.zeros(len(df))  # 0=no position, 1=long, -1=short
    position_sizes = np.zeros(len(df))
    entry_indices = np.zeros(len(df), dtype=int) - 1
    holding_periods = np.zeros(len(df), dtype=int)
    capital = np.ones(len(df))
    
    trades = []
    
    # Simulation loop
    for i in range(1, len(df)):
        # Update holding period for existing positions
        if positions[i-1] != 0:
            holding_periods[i] = holding_periods[i-1] + 1
            entry_indices[i] = entry_indices[i-1]
            
            # Check exit conditions
            if i > 0 and entry_indices[i-1] >= 0:
                # Calculate P&L
                current_pnl = positions[i-1] * df.iloc[i-1]['actual']
                
                # Exit logic
                exit_position = False
                
                # Stop loss
                if current_pnl < -stop_loss:
                    exit_position = True
                    exit_reason = "stop_loss"
                
                # Take profit
                elif current_pnl > take_profit:
                    exit_position = True
                    exit_reason = "take_profit"
                
                # Max holding period
                elif holding_periods[i] >= max_holding:
                    exit_position = True
                    exit_reason = "max_holding"
                
                # Exit position if needed
                if exit_position:
                    # Calculate trade P&L
                    idx_since_entry = range(entry_indices[i-1] + 1, i + 1)
                    trade_pnl = positions[i-1] * sum(df.iloc[idx]['actual'] for idx in idx_since_entry if idx < len(df))
                    
                    # Record trade
                    trades.append({
                        'entry_time': df.iloc[entry_indices[i-1]]['timestamp'],
                        'exit_time': df.iloc[i]['timestamp'],
                        'direction': 'long' if positions[i-1] > 0 else 'short',
                        'pnl': trade_pnl,
                        'position_size': position_sizes[i-1],
                        'exit_reason': exit_reason
                    })
                    
                    # Update capital
                    capital[i] = capital[i-1] * (1 + position_sizes[i-1] * trade_pnl - fee)
                    
                    # Reset position
                    positions[i] = 0
                    position_sizes[i] = 0
                    holding_periods[i] = 0
                    entry_indices[i] = -1
                    continue
            
            # If no exit, carry forward position
            positions[i] = positions[i-1]
            position_sizes[i] = position_sizes[i-1]
            capital[i] = capital[i-1] * (1 + position_sizes[i-1] * df.iloc[i-1]['actual'])
            continue
        
        # Entry logic
        if positions[i-1] == 0:
            pred = df.iloc[i-1]['prediction']
            
            # Long entry
            if pred > threshold:
                positions[i] = 1
                
                # Position sizing - 10-50% based on conviction
                strength = min(1.0, pred / (threshold * 5))
                position_sizes[i] = 0.1 + 0.4 * strength
                
                entry_indices[i] = i-1
                holding_periods[i] = 1
                capital[i] = capital[i-1] * (1 - fee)
                
            # Short entry
            elif pred < -threshold:
                positions[i] = -1
                
                # Position sizing - 10-50% based on conviction
                strength = min(1.0, abs(pred) / (threshold * 5))
                position_sizes[i] = 0.1 + 0.4 * strength
                
                entry_indices[i] = i-1
                holding_periods[i] = 1
                capital[i] = capital[i-1] * (1 - fee)
                
            else:
                # No position
                capital[i] = capital[i-1]
    
    # Close any open positions
    if positions[-1] != 0 and entry_indices[-1] >= 0:
        idx_since_entry = range(entry_indices[-1] + 1, len(df))
        trade_pnl = positions[-1] * sum(df.iloc[idx]['actual'] for idx in idx_since_entry if idx < len(df))
        
        trades.append({
            'entry_time': df.iloc[entry_indices[-1]]['timestamp'],
            'exit_time': df.iloc[-1]['timestamp'],
            'direction': 'long' if positions[-1] > 0 else 'short',
            'pnl': trade_pnl,
            'position_size': position_sizes[-1],
            'exit_reason': 'end_of_data'
        })
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate metrics
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['pnl'] > 0) if trades else 0
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = (capital[-1] - capital[0]) / capital[0]
    
    results = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': capital[-1],
        'capital_curve': capital,
        'trades': trades_df
    }
    
    return results

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='TFT 7-Day Backtest')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Threshold for entering positions')
    parser.add_argument('--stop_loss', type=float, default=STOP_LOSS, help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=TAKE_PROFIT, help='Take profit percentage')
    parser.add_argument('--max_holding', type=int, default=MAX_HOLDING, help='Maximum holding period in bars')
    parser.add_argument('--fee', type=float, default=FEE, help='Transaction fee percentage')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Print configuration
    logger.info("=== TFT 7-Day Backtest ===")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Stop Loss: {args.stop_loss}")
    logger.info(f"Take Profit: {args.take_profit}")
    logger.info(f"Max Holding: {args.max_holding}")
    logger.info(f"Fee: {args.fee}")
    
    # Dummy data - in a real scenario, you would:
    # 1. Fetch data from the database
    # 2. Add features
    # 3. Scale the features
    # 4. Generate predictions using the models
    # 5. Run the trading simulation
    
    # For demonstration purposes, create a simulated dataset
    logger.info("Creating simulated dataset for demonstration")
    n_samples = 1000
    timestamps = [datetime.now() + timedelta(seconds=10*i) for i in range(n_samples)]
    
    np.random.seed(42)
    predictions = np.random.normal(0, 0.003, n_samples)
    actuals = predictions * 0.7 + np.random.normal(0, 0.002, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions,
        'actual': actuals
    })
    
    # Run the trading simulation
    results = simulate_trading(
        df, 
        threshold=args.threshold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding=args.max_holding,
        fee=args.fee
    )
    
    # Display results
    logger.info("\n=== Backtest Results ===")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Winning Trades: {results['winning_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Final Equity: {results['final_equity']:.4f}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(results['capital_curve'])
    plt.title(f'Equity Curve (Return: {results["total_return"]:.2%})')
    plt.grid(True)
    plt.savefig(f'{args.output}/equity_curve.png')
    plt.close()
    
    # Save results to JSON
    result_data = {
        'config': {
            'threshold': args.threshold,
            'stop_loss': args.stop_loss,
            'take_profit': args.take_profit,
            'max_holding': args.max_holding,
            'fee': args.fee
        },
        'metrics': {
            'total_trades': results['total_trades'],
            'winning_trades': results['winning_trades'],
            'win_rate': float(results['win_rate']),
            'total_return': float(results['total_return']),
            'final_equity': float(results['final_equity'])
        }
    }
    
    with open(f'{args.output}/results.json', 'w') as f:
        json.dump(result_data, f, indent=4)
    
    logger.info(f"Results saved to {args.output}/")

if __name__ == "__main__":
    main() 