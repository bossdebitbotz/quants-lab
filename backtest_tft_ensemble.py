import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Constants
DIRECTIONAL_MODEL_PATH = 'path_to_directional_model'
DOWNWARD_MODEL_PATH = 'path_to_downward_model'
ENSEMBLE_THRESHOLD = 0.05
STOP_LOSS = 0.02
TAKE_PROFIT = 0.05
MAX_HOLDING_PERIOD = 10
TRANSACTION_FEE = 0.001

# Setup logging
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Add derived features for TFT model
    
    Args:
        df: DataFrame with orderbook data
        
    Returns:
        DataFrame with additional features
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Calculate price returns at different timeframes
    df_features['returns_10sec'] = df_features['mid_price'].pct_change(1)
    df_features['returns_30sec'] = df_features['mid_price'].pct_change(3)
    df_features['returns_1min'] = df_features['mid_price'].pct_change(6)
    
    # Calculate imbalance features
    df_features['buy_sell_imbalance'] = (df_features['bid_quantity'] - df_features['ask_quantity']) / (df_features['bid_quantity'] + df_features['ask_quantity'])
    
    # Add volatility feature (rolling standard deviation of returns)
    df_features['volatility_1min'] = df_features['returns_10sec'].rolling(window=6).std()
    
    # Create order count features (approximations based on available data)
    # These are placeholder features that would ideally come from more detailed data
    df_features['new_bid_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['new_ask_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['canceled_bid_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['canceled_ask_orders'] = np.random.randint(0, 10, size=len(df_features))
    df_features['executed_bid_orders'] = np.random.randint(0, 5, size=len(df_features))
    df_features['executed_ask_orders'] = np.random.randint(0, 5, size=len(df_features))
    
    # Create the target variable (next period return)
    df_features['target'] = df_features['returns_10sec'].shift(-1)
    
    # Create MA-adjusted target (target - rolling_mean)
    window_size = 30
    df_features['target_ma'] = df_features['target'].rolling(window=window_size).mean()
    df_features['target_ma_adjusted'] = df_features['target'] - df_features['target_ma']
    
    # Fill NaN values with 0
    df_features = df_features.fillna(0)
    
    return df_features 

def main():
    """Main function to run the backtest"""
    parser = argparse.ArgumentParser(description='TFT Ensemble Model Backtester')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use (default: 7)')
    parser.add_argument('--directional-model', type=str, default=DIRECTIONAL_MODEL_PATH, help='Path to directional model')
    parser.add_argument('--downward-model', type=str, default=DOWNWARD_MODEL_PATH, help='Path to downward specialist model')
    parser.add_argument('--threshold', type=float, default=ENSEMBLE_THRESHOLD, help='Base threshold for trade entry')
    parser.add_argument('--stop-loss', type=float, default=STOP_LOSS, help='Stop loss percentage') 
    parser.add_argument('--take-profit', type=float, default=TAKE_PROFIT, help='Take profit percentage')
    parser.add_argument('--max-holding', type=int, default=MAX_HOLDING_PERIOD, help='Maximum holding period')
    parser.add_argument('--fee', type=float, default=TRANSACTION_FEE, help='Transaction fee per trade')
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
    logger.info(f"Data Period: {args.days} days")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Stop Loss: {args.stop_loss}")
    logger.info(f"Take Profit: {args.take_profit}")
    logger.info(f"Max Holding Period: {args.max_holding}")
    logger.info(f"Transaction Fee: {args.fee}")
    logger.info(f"Position Sizing: {'Enabled' if args.position_sizing else 'Disabled'}")
    logger.info(f"Dynamic Threshold: {'Enabled' if args.dynamic_threshold else 'Disabled'}")
    
    # Step 1: Load order book data from the database
    logger.info(f"Loading {args.days} days of orderbook data from database")
    df = load_orderbook_data(days=args.days)
    if df is None or len(df) < 100:
        logger.error("Insufficient data to perform backtest")
        return
    
    # Step 2: Add derived features
    df_features = add_derived_features(df)
    
    # Step 3: Prepare data for TFT model
    X_tensor, y_array, timestamps = prepare_data_for_tft(
        df_features, 
        target_column=args.target_col,
        context_length=args.context_length
    )
    
    # Step 4: Load models
    directional_model = load_model(args.directional_model)
    downward_model = load_model(args.downward_model)
    
    if directional_model is None or downward_model is None:
        logger.error("Failed to load models")
        return
    
    # Step 5: Generate ensemble predictions
    ensemble_preds = generate_ensemble_predictions(
        X_tensor,
        directional_model,
        downward_model
    )
    
    # Step 6: Run trading simulation
    trading_results = simulate_trading(
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
    plot_results(trading_results, output_dir=args.output_dir)
    
    # Step 8: Display summary
    logger.info("\n=== Backtest Results ===")
    logger.info(f"Total Trades: {trading_results['total_trades']}")
    logger.info(f"Win Rate: {trading_results['win_rate']:.2%}")
    logger.info(f"Total Return: {trading_results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.2f}")
    logger.info(f"Final Equity: {trading_results['final_equity']:.4f}")
    logger.info(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 