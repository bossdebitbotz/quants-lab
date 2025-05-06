#!/usr/bin/env python3
"""
Run TFT Backtest

This script runs the TFT backtest with various configuration options.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the backtester
from tft_orderbook_backtest import TFTBacktester

# Import TFT bot modules directly using sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'realtime_tft_bot'))
import config as tft_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_backtest")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run TFT backtest')
    
    parser.add_argument('--trading-pair', type=str, default='WLD-USDT',
                        help='Trading pair to backtest (default: WLD-USDT)')
    
    parser.add_argument('--start-date', type=str, 
                        help='Start date for backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                        help='End date for backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to backtest if no start/end date provided (default: 7)')
    
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for backtest (default: 10000)')
    
    parser.add_argument('--trade-threshold', type=float, default=0.0005,
                        help='Threshold for trading signals (default: 0.0005)')
    
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as percentage of balance (default: 0.1)')
    
    parser.add_argument('--take-profit', type=float, default=0.01,
                        help='Take profit percentage (default: 0.01)')
    
    parser.add_argument('--stop-loss', type=float, default=0.005,
                        help='Stop loss percentage (default: 0.005)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Load the configuration from the TFT bot
    config_dict = tft_config.load_config()
    
    # Add backtest specific configurations
    config_dict['BACKTEST_INITIAL_BALANCE'] = args.initial_balance
    config_dict['BACKTEST_TRADE_THRESHOLD'] = args.trade_threshold
    config_dict['BACKTEST_POSITION_SIZE_PCT'] = args.position_size
    config_dict['BACKTEST_TAKE_PROFIT_PCT'] = args.take_profit
    config_dict['BACKTEST_STOP_LOSS_PCT'] = args.stop_loss
    
    # Create backtester instance
    backtester = TFTBacktester(config_dict, trading_pair=args.trading_pair)
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') + timedelta(days=1)  # Include full end day
    else:
        # Find available data range from database
        conn = backtester.connect_to_db()
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        MIN(DATE_TRUNC('day', timestamp)) as first_date,
                        MAX(DATE_TRUNC('day', timestamp)) as last_date
                    FROM order_events
                    WHERE trading_pair = %s
                """, (args.trading_pair,))
                
                result = cursor.fetchone()
                
            if result and result[0] and result[1]:
                first_date = result[0]
                last_date = result[1]
                
                # If no dates specified, use last N days
                if not args.start_date and not args.end_date:
                    end_date = last_date + timedelta(days=1)  # Include full end day
                    start_date = end_date - timedelta(days=args.days)
                    
                    # Ensure start_date is not before the first available date
                    if start_date < first_date:
                        start_date = first_date
                
                logger.info(f"Available data range: {first_date} to {last_date}")
                logger.info(f"Selected backtest range: {start_date} to {end_date}")
            else:
                logger.error("Could not determine data date range.")
                return
                
        except Exception as e:
            logger.error(f"Error determining date range: {type(e).__name__} - {str(e)}")
            return
        finally:
            if conn:
                conn.close()
    
    # Run the backtest with the specified date range
    backtester = TFTBacktester(
        config=config_dict,
        trading_pair=args.trading_pair,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Running backtest for {args.trading_pair} from {start_date} to {end_date}")
    logger.info(f"Configuration: initial_balance={args.initial_balance}, trade_threshold={args.trade_threshold}, "
                f"position_size={args.position_size}, take_profit={args.take_profit}, stop_loss={args.stop_loss}")
    
    results = backtester.run_backtest()
    
    if results:
        logger.info("Backtest completed successfully. Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        
        logger.info(f"Results and plots saved to backtesting/ directory")
    else:
        logger.error("Backtest failed to produce results.")

if __name__ == "__main__":
    main() 