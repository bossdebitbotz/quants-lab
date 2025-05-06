#!/usr/bin/env python3
"""
TFT Bot Backtester using Order Book Data

This script backtests the TFT bot against real order book data collected
by the order_lifecycle_tracker. It uses the exact same feature calculation
and prediction pipeline as the live bot.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add the realtime_tft_bot directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'realtime_tft_bot'))

# Import TFT bot modules
from utils.feature_calculator import FeatureCalculator
from utils.tft_predictor import TFTPredictor
import config as tft_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tft_backtest.log"),
        logging.StreamHandler()
    ]
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

class TFTBacktester:
    """
    Backtests the TFT bot against historical order book data
    """
    
    def __init__(self, config, trading_pair='WLD-USDT', start_date=None, end_date=None):
        """
        Initialize the backtester
        
        Args:
            config: Configuration dictionary for the TFT bot
            trading_pair: The trading pair to backtest on
            start_date: Start date for backtest (datetime or string)
            end_date: End date for backtest (datetime or string)
        """
        self.config = config
        self.trading_pair = trading_pair
        
        # Parse dates if provided as strings
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        else:
            self.end_date = end_date
        
        # Initialize feature calculator and TFT predictor
        self.feature_calculator = FeatureCalculator(config)
        self.feature_calculator.initialize()
        
        self.tft_predictor = TFTPredictor(config)
        self.tft_predictor.initialize()
        
        # Portfolio tracking
        self.initial_balance = config.get('BACKTEST_INITIAL_BALANCE', 10000)
        self.position = 0  # Current position in base currency
        self.balance = self.initial_balance  # Current balance in quote currency
        
        # Trading parameters
        self.trade_threshold = config.get('BACKTEST_TRADE_THRESHOLD', 0.0005)
        self.position_size_pct = config.get('BACKTEST_POSITION_SIZE_PCT', 0.1)
        self.take_profit_pct = config.get('BACKTEST_TAKE_PROFIT_PCT', 0.01)
        self.stop_loss_pct = config.get('BACKTEST_STOP_LOSS_PCT', 0.005)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.predictions = []
    
    def connect_to_db(self):
        """Connect to the PostgreSQL database"""
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
            logger.error(f"Error connecting to database: {type(e).__name__} - {str(e)}")
            return None
    
    def get_ohlcv_data(self, interval_minutes=1):
        """
        Retrieve OHLCV data from order book events
        
        This reconstructs OHLCV bars from the order events table by finding
        the mid-price at each timestamp.
        
        Args:
            interval_minutes: Interval in minutes for the OHLCV bars
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Retrieving OHLCV data at {interval_minutes}-minute intervals")
        
        try:
            conn = self.connect_to_db()
            
            # Query to reconstruct OHLCV from order book data
            query = """
            WITH time_series AS (
                SELECT 
                    generate_series(
                        DATE_TRUNC('minute', MIN(timestamp)),
                        DATE_TRUNC('minute', MAX(timestamp)),
                        %s::interval
                    ) AS bar_time
                FROM order_events
                WHERE trading_pair = %s
                    AND timestamp BETWEEN %s AND %s
            ),
            price_samples AS (
                SELECT
                    ts.bar_time,
                    e.timestamp,
                    e.price,
                    e.side,
                    ROW_NUMBER() OVER (PARTITION BY ts.bar_time, e.side ORDER BY e.timestamp DESC) as rn
                FROM time_series ts
                JOIN order_events e ON e.timestamp <= ts.bar_time + %s::interval
                WHERE e.trading_pair = %s
                    AND e.timestamp >= ts.bar_time - %s::interval
                    AND e.timestamp BETWEEN %s AND %s
            ),
            bar_prices AS (
                SELECT
                    bar_time,
                    AVG(CASE WHEN side = 'BID' THEN price END) as bid_price,
                    AVG(CASE WHEN side = 'ASK' THEN price END) as ask_price,
                    COUNT(*) as sample_count
                FROM price_samples
                WHERE rn <= 10  -- Take top 10 most recent prices per side
                GROUP BY bar_time
            ),
            volume_data AS (
                SELECT
                    DATE_TRUNC('minute', timestamp) AS trunc_time,
                    SUM(quantity) AS volume
                FROM order_events
                WHERE trading_pair = %s
                    AND event_type IN ('FILL', 'CANCEL')
                    AND timestamp BETWEEN %s AND %s
                GROUP BY trunc_time
            )
            SELECT
                bp.bar_time as timestamp,
                FIRST_VALUE(bp.bid_price + bp.ask_price) OVER (PARTITION BY bp.bar_time ORDER BY bp.bar_time) / 2 as open,
                MAX((bp.bid_price + bp.ask_price) / 2) OVER (PARTITION BY bp.bar_time) as high,
                MIN((bp.bid_price + bp.ask_price) / 2) OVER (PARTITION BY bp.bar_time) as low,
                LAST_VALUE(bp.bid_price + bp.ask_price) OVER (PARTITION BY bp.bar_time ORDER BY bp.bar_time ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) / 2 as close,
                COALESCE(vd.volume, 0) as volume
            FROM bar_prices bp
            LEFT JOIN volume_data vd ON bp.bar_time = vd.trunc_time
            WHERE bp.sample_count > 0
            ORDER BY bp.bar_time
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    f"{interval_minutes} minutes",  # Bar interval
                    self.trading_pair,
                    self.start_date, 
                    self.end_date,
                    f"{interval_minutes} minutes",  # Lookback interval for price
                    self.trading_pair,
                    f"{interval_minutes} minutes",  # Lookback interval for price
                    self.start_date,
                    self.end_date,
                    self.trading_pair,
                    self.start_date,
                    self.end_date
                ))
                
                # Fetch results
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
            # Convert to DataFrame
            df = pd.DataFrame(results, columns=columns)
            
            # Convert timestamp column to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} OHLCV bars")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {type(e).__name__} - {str(e)}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def get_order_book_snapshot(self, timestamp):
        """
        Get order book snapshot at a specific timestamp
        
        Args:
            timestamp: The timestamp to get the snapshot for
            
        Returns:
            Dict with bids and asks
        """
        try:
            conn = self.connect_to_db()
            
            # Query to get the order book snapshot
            query = """
            WITH order_status AS (
                SELECT 
                    synthetic_order_id,
                    price,
                    quantity,
                    side,
                    MAX(timestamp) as latest_timestamp
                FROM order_events
                WHERE trading_pair = %s
                    AND timestamp <= %s
                GROUP BY synthetic_order_id, price, quantity, side
            ),
            latest_events AS (
                SELECT 
                    oe.synthetic_order_id,
                    oe.price,
                    oe.quantity,
                    oe.side,
                    oe.event_type
                FROM order_events oe
                JOIN order_status os ON oe.synthetic_order_id = os.synthetic_order_id AND oe.timestamp = os.latest_timestamp
                WHERE oe.trading_pair = %s
                    AND oe.timestamp <= %s
            )
            SELECT 
                price,
                side,
                SUM(quantity) as total_quantity
            FROM latest_events
            WHERE event_type NOT IN ('CANCEL', 'FILL')
            GROUP BY price, side
            ORDER BY 
                CASE WHEN side = 'BID' THEN -price ELSE price END
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    self.trading_pair,
                    timestamp,
                    self.trading_pair,
                    timestamp
                ))
                
                # Fetch results
                results = cursor.fetchall()
            
            # Process results into order book format
            order_book = {
                'bids': {},
                'asks': {}
            }
            
            for price, side, quantity in results:
                if side == 'BID':
                    order_book['bids'][str(price)] = {'quantity': float(quantity)}
                else:
                    order_book['asks'][str(price)] = {'quantity': float(quantity)}
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error retrieving order book snapshot: {type(e).__name__} - {str(e)}")
            return {'bids': {}, 'asks': {}}
        finally:
            if conn:
                conn.close()
    
    def execute_trade(self, timestamp, price, direction, amount):
        """
        Execute a trade in the backtest
        
        Args:
            timestamp: Time of the trade
            price: Execution price
            direction: 'BUY' or 'SELL'
            amount: Amount to trade in quote currency
            
        Returns:
            Dict with trade details
        """
        # Calculate quantity and update balance
        quantity = amount / price
        
        trade = {
            'timestamp': timestamp,
            'price': price,
            'direction': direction,
            'amount': amount,
            'quantity': quantity,
            'balance_before': self.balance,
            'position_before': self.position
        }
        
        if direction == 'BUY':
            self.balance -= amount
            self.position += quantity
        else:  # SELL
            self.balance += amount
            self.position -= quantity
        
        trade['balance_after'] = self.balance
        trade['position_after'] = self.position
        
        # Calculate portfolio value
        trade['portfolio_value'] = self.balance + (self.position * price)
        
        # Add to trades list
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(self):
        """
        Run the backtest
        
        Returns:
            Dict with backtest results
        """
        logger.info(f"Starting backtest for {self.trading_pair} from {self.start_date} to {self.end_date}")
        
        # Get OHLCV data
        ohlcv_df = self.get_ohlcv_data(interval_minutes=1)
        if ohlcv_df.empty:
            logger.error("No OHLCV data retrieved. Backtest cannot run.")
            return None
        
        # Fix dataframe types to ensure they're properly processed
        ohlcv_df = self.fix_dataframe_types(ohlcv_df)
        
        # Print a sample of the data to verify
        logger.info(f"OHLCV data sample:")
        logger.info(f"Columns: {ohlcv_df.columns.tolist()}")
        logger.info(f"Types: {ohlcv_df.dtypes}")
        logger.info(f"First 3 rows: \n{ohlcv_df.head(3)}")
        
        # Initialize feature calculator with first set of data
        try:
            initial_data = ohlcv_df.iloc[:self.config['CONTEXT_LENGTH']].reset_index()
            initial_data.rename(columns={'timestamp': 'time'}, inplace=True)
            
            # Ensure the necessary columns exist and are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in initial_data.columns:
                    logger.error(f"Required column '{col}' not found in OHLCV data")
                    return None
                
            # Log the initial data being sent to the feature calculator
            logger.info(f"Initial data shape for feature calculator: {initial_data.shape}")
            logger.info(f"Initial data columns: {initial_data.columns.tolist()}")
            
            # Update with initial data
            update_result = self.feature_calculator.update_with_ohlcv(initial_data)
            if not update_result:
                logger.error("Failed to initialize feature calculator with initial data")
                return None
        except Exception as e:
            logger.error(f"Error initializing feature calculator: {type(e).__name__} - {str(e)}")
            return None
        
        # Initialize tracking variables
        last_trade_price = None
        in_position = False
        take_profit_level = None
        stop_loss_level = None
        
        # Main backtest loop
        logger.info("Running backtest...")
        for i in tqdm(range(self.config['CONTEXT_LENGTH'], len(ohlcv_df))):
            timestamp = ohlcv_df.index[i]
            current_price = ohlcv_df.iloc[i]['close']
            
            # Get current bar data
            current_bar = ohlcv_df.iloc[i:i+1].reset_index()
            current_bar.rename(columns={'timestamp': 'time'}, inplace=True)
            
            # Update feature calculator with new bar
            self.feature_calculator.update_with_ohlcv(current_bar)
            
            # Get order book at this timestamp
            order_book = self.get_order_book_snapshot(timestamp)
            
            # Add order book features
            self.feature_calculator.add_order_book_features(order_book)
            
            # Prepare model input
            model_input = self.feature_calculator.prepare_model_input()
            
            if model_input is None:
                logger.warning(f"Could not prepare model input at {timestamp}. Skipping.")
                continue
            
            # Get TFT prediction
            prediction = self.tft_predictor.predict(model_input)
            
            # Save prediction
            self.predictions.append({
                'timestamp': timestamp,
                'price': current_price,
                'dir_prediction': prediction['dir_prediction'],
                'down_prediction': prediction['down_prediction'],
                'ensemble_prediction': prediction['ensemble_prediction']
            })
            
            # Trading logic
            signal = prediction['ensemble_prediction']
            
            # Check take-profit and stop-loss if we're in a position
            if in_position:
                # Check for exit conditions
                if last_trade_price is not None:
                    # Take profit
                    if current_price >= take_profit_level and self.position > 0:
                        logger.info(f"Take profit hit at {timestamp}: {current_price} >= {take_profit_level}")
                        trade_amount = self.position * current_price
                        self.execute_trade(timestamp, current_price, 'SELL', trade_amount)
                        in_position = False
                        last_trade_price = None
                    
                    # Stop loss
                    elif current_price <= stop_loss_level and self.position > 0:
                        logger.info(f"Stop loss hit at {timestamp}: {current_price} <= {stop_loss_level}")
                        trade_amount = self.position * current_price
                        self.execute_trade(timestamp, current_price, 'SELL', trade_amount)
                        in_position = False
                        last_trade_price = None
            
            # Entry signal
            elif not in_position and abs(signal) > self.trade_threshold:
                direction = 'BUY' if signal > 0 else 'SELL'
                
                # For simplicity, we'll only take long positions in this backtest
                if direction == 'BUY':
                    trade_amount = self.balance * self.position_size_pct
                    logger.info(f"BUY signal at {timestamp}: {signal} > {self.trade_threshold}")
                    
                    self.execute_trade(timestamp, current_price, direction, trade_amount)
                    in_position = True
                    last_trade_price = current_price
                    
                    # Set take-profit and stop-loss levels
                    take_profit_level = current_price * (1 + self.take_profit_pct)
                    stop_loss_level = current_price * (1 - self.stop_loss_pct)
            
            # Update equity curve
            portfolio_value = self.balance + (self.position * current_price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'price': current_price,
                'balance': self.balance,
                'position': self.position,
                'portfolio_value': portfolio_value,
                'signal': signal
            })
        
        # Calculate performance metrics
        results = self.calculate_performance()
        
        # Plot results
        self.plot_results()
        
        return results
    
    def calculate_performance(self):
        """
        Calculate backtest performance metrics
        
        Returns:
            Dict with performance metrics
        """
        if not self.equity_curve:
            return {
                'total_return_pct': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['return'] = equity_df['portfolio_value'].pct_change()
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate metrics
        start_value = self.initial_balance
        end_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (end_value - start_value) / start_value
        
        # Duration in years
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
        
        # Daily returns for Sharpe ratio
        equity_df['daily_return'] = equity_df['portfolio_value'].resample('D').last().pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['cummax'] - equity_df['portfolio_value']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].max()
        
        # Win rate
        if not trades_df.empty and len(trades_df) > 1:
            trades_df['profit'] = trades_df['portfolio_value'] - trades_df['portfolio_value'].shift(1)
            winning_trades = (trades_df['profit'] > 0).sum()
            total_trades = len(trades_df) - 1  # Subtract 1 to account for the shift
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        # Save results to file
        results = {
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate * 100 if win_rate > 0 else 0,
            'total_trades': total_trades,
            'start_value': start_value,
            'end_value': end_value,
            'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log results
        logger.info("Backtest Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        
        # Save detailed results to CSV
        equity_df.to_csv('backtesting/equity_curve.csv')
        
        if not trades_df.empty:
            trades_df.to_csv('backtesting/trades.csv')
        
        pd.DataFrame(self.predictions).to_csv('backtesting/predictions.csv')
        
        # Save results to JSON
        with open('backtesting/backtest_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.equity_curve:
            logger.warning("No data to plot")
            return
        
        logger.info("Generating plots...")
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(self.predictions)
        predictions_df.set_index('timestamp', inplace=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot price
        axes[0].plot(equity_df.index, equity_df['price'], label='Price')
        axes[0].set_title(f'{self.trading_pair} Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot portfolio value
        axes[1].plot(equity_df.index, equity_df['portfolio_value'], label='Portfolio Value')
        axes[1].set_title('Portfolio Value')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot signals and trades
        axes[2].plot(equity_df.index, equity_df['signal'], label='Signal', color='purple', alpha=0.7)
        axes[2].axhline(y=self.trade_threshold, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        axes[2].axhline(y=-self.trade_threshold, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        
        # Add buy/sell markers if there are trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            
            # Add buy trades
            buy_trades = trades_df[trades_df['direction'] == 'BUY']
            if not buy_trades.empty:
                axes[0].scatter(
                    buy_trades['timestamp'], 
                    buy_trades['price'], 
                    marker='^', 
                    color='green', 
                    s=100, 
                    label='Buy'
                )
            
            # Add sell trades
            sell_trades = trades_df[trades_df['direction'] == 'SELL']
            if not sell_trades.empty:
                axes[0].scatter(
                    sell_trades['timestamp'], 
                    sell_trades['price'], 
                    marker='v', 
                    color='red', 
                    s=100, 
                    label='Sell'
                )
            
            # Update legend
            axes[0].legend()
        
        axes[2].set_title('Trading Signals')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('backtesting/backtest_results.png')
        logger.info("Plots saved to 'backtesting/backtest_results.png'")
        
        # Also create a predictions vs actual plot
        if not predictions_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot price
            ax.plot(predictions_df.index, predictions_df['price'], label='Price', color='blue')
            
            # Create a secondary y-axis for predictions
            ax2 = ax.twinx()
            ax2.plot(predictions_df.index, predictions_df['ensemble_prediction'], label='Ensemble Prediction', color='purple', alpha=0.7)
            ax2.plot(predictions_df.index, predictions_df['dir_prediction'], label='Directional Prediction', color='green', alpha=0.5)
            ax2.plot(predictions_df.index, predictions_df['down_prediction'], label='Downward Prediction', color='red', alpha=0.5)
            
            ax.set_title(f'{self.trading_pair} Price vs Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax2.set_ylabel('Prediction Value')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax.grid(True)
            plt.tight_layout()
            plt.savefig('backtesting/predictions_vs_actual.png')
            logger.info("Predictions plot saved to 'backtesting/predictions_vs_actual.png'")

    def fix_dataframe_types(self, df):
        """Ensure dataframe columns have the correct types for feature calculation"""
        if df is None or df.empty:
            return df
        
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

def test_database_connection():
    """Test database connection and query data availability"""
    try:
        # Create DB connection
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        with conn.cursor() as cursor:
            # Check if order_events table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'order_events'
                );
            """)
            table_exists = cursor.fetchone()[0]
            logger.info(f"Order events table exists: {table_exists}")
            
            if table_exists:
                # Check count of records
                cursor.execute("SELECT COUNT(*) FROM order_events;")
                total_count = cursor.fetchone()[0]
                logger.info(f"Total order events: {total_count}")
                
                # Check WLD-USDT records
                cursor.execute("SELECT COUNT(*) FROM order_events WHERE trading_pair = 'WLD-USDT';")
                pair_count = cursor.fetchone()[0]
                logger.info(f"WLD-USDT order events: {pair_count}")
                
                # Check date range
                cursor.execute("""
                    SELECT 
                        MIN(DATE_TRUNC('day', timestamp)),
                        MAX(DATE_TRUNC('day', timestamp))
                    FROM order_events
                    WHERE trading_pair = 'WLD-USDT';
                """)
                date_range = cursor.fetchone()
                logger.info(f"WLD-USDT date range: {date_range[0]} to {date_range[1]}")
                
                # Check sample data
                cursor.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM order_events 
                    WHERE trading_pair = 'WLD-USDT'
                    GROUP BY event_type;
                """)
                event_counts = cursor.fetchall()
                logger.info("Event type counts:")
                for event_type, count in event_counts:
                    logger.info(f"  {event_type}: {count}")
                
        return True
    except Exception as e:
        logger.error(f"Database test failed: {type(e).__name__} - {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def main():
    """Main function"""
    # Load the configuration from the TFT bot
    config_dict = tft_config.load_config()
    
    # Add backtest specific configurations
    config_dict['BACKTEST_INITIAL_BALANCE'] = 10000
    config_dict['BACKTEST_TRADE_THRESHOLD'] = 0.0005
    config_dict['BACKTEST_POSITION_SIZE_PCT'] = 0.1
    config_dict['BACKTEST_TAKE_PROFIT_PCT'] = 0.01
    config_dict['BACKTEST_STOP_LOSS_PCT'] = 0.005
    
    # Find the start and end dates of available data
    backtester = TFTBacktester(config_dict)
    conn = backtester.connect_to_db()
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    MIN(DATE_TRUNC('day', timestamp)) as start_date,
                    MAX(DATE_TRUNC('day', timestamp)) as end_date
                FROM order_events
                WHERE trading_pair = 'WLD-USDT'
            """)
            
            result = cursor.fetchone()
            
        if result and result[0] and result[1]:
            start_date = result[0]
            end_date = result[1] + timedelta(days=1)  # Include the full end day
            
            logger.info(f"Available data range: {start_date} to {end_date}")
            
            # Run the backtest
            backtester = TFTBacktester(
                config=config_dict,
                trading_pair='WLD-USDT',
                start_date=start_date,
                end_date=end_date
            )
            
            results = backtester.run_backtest()
            
            if results:
                logger.info("Backtest completed successfully. Results:")
                for key, value in results.items():
                    logger.info(f"{key}: {value}")
            else:
                logger.error("Backtest failed to produce results.")
                
        else:
            logger.error("Could not determine data date range.")
            
    except Exception as e:
        logger.error(f"Error in main function: {type(e).__name__} - {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Test the database connection first
    if test_database_connection():
        main()
    else:
        logger.error("Database test failed. Please check database connection and data availability.") 