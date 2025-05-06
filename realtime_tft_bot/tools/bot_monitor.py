#!/usr/bin/env python3
"""
TFT Bot Monitoring UI

This script provides a command-line interface for monitoring the TFT trading bot.
It displays real-time information about positions, trades, model predictions,
and system status in a consolidated view.
"""

import os
import sys
import asyncio
import logging
import argparse
import curses
import time
import subprocess
import pandas as pd
from datetime import datetime, timedelta

# Configure logging first, in case imports fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="bot_monitor.log",
    filemode="a"
)

logger = logging.getLogger("BotMonitor")

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Try to import from the parent directory
try:
    from utils.db_logger import DBLogger
    from utils.pnl_tracker import calculate_pnl_stats  # Import the standardized PnL tracker
    from config import load_config
    logger.info("Successfully imported modules from parent directory")
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    print(f"Error: {str(e)}")
    print("Ensure you've run setup_monitor.sh to install dependencies and create necessary links")
    sys.exit(1)

class TFTBotMonitor:
    """Command-line UI for monitoring the TFT trading bot"""
    
    def __init__(self):
        """Initialize the monitor"""
        self.config = load_config()
        self.db_logger = None
        self.trading_pair = self.config.get('TRADING_PAIR', 'Unknown')
        self.position_state = None
        self.recent_trades = []
        self.recent_predictions = []
        self.pnl_stats = {}
        self.error_messages = []
        self.last_updated = datetime.now()
        self.is_running = True
        self.refresh_interval = 5  # seconds

    async def initialize(self):
        """Initialize the database connection"""
        self.db_logger = DBLogger(self.config)
        if not await self.db_logger.initialize():
            logger.error("Failed to initialize database connection")
            return False
        return True

    async def close(self):
        """Clean up resources"""
        if self.db_logger:
            await self.db_logger.close()

    async def fetch_current_position(self):
        """Fetch the current position state"""
        if not self.db_logger:
            return None
        
        try:
            position_state = await self.db_logger.get_position_state(self.trading_pair)
            self.position_state = position_state
            
            # If we have a position, get current market price to calculate unrealized PnL
            if position_state and position_state['position'] != 'NONE':
                # Get the latest price either from recent trades or executed_trades
                latest_price = await self.get_latest_price()
                
                if latest_price and position_state['entry_price']:
                    # Calculate unrealized PnL
                    try:
                        entry_price = float(position_state['entry_price'])
                        position_size = float(position_state['position_size'])
                        
                        if position_state['position'] == 'LONG':
                            # For long: unrealized_pnl = (current_price - entry_price) * size
                            unrealized_pnl = (latest_price - entry_price) * position_size
                        else:  # SHORT
                            # For short: unrealized_pnl = (entry_price - current_price) * size
                            unrealized_pnl = (entry_price - latest_price) * position_size
                            
                        # Add to position state
                        position_state['current_price'] = latest_price
                        position_state['unrealized_pnl'] = unrealized_pnl
                        position_state['pnl_percent'] = (unrealized_pnl / (entry_price * position_size)) * 100
                        
                        logger.info(f"Calculated unrealized PnL: {unrealized_pnl:.6f} ({position_state['pnl_percent']:.2f}%)")
                    except (ValueError, TypeError, ZeroDivisionError) as e:
                        logger.warning(f"Error calculating unrealized PnL: {e}")
            
            return position_state
        except Exception as e:
            logger.error(f"Error fetching position state: {str(e)}")
            self.add_error(f"Failed to fetch position: {str(e)}")
            return None
            
    async def get_latest_price(self):
        """Get the latest market price from recent trades or predictions"""
        if not self.db_logger:
            return None
            
        try:
            conn = self.db_logger._get_connection()
            if not conn:
                logger.error("Failed to get database connection")
                return None
                
            # Extract the trading pair components for different formats
            pair_components = self.trading_pair.split('-')
            exchange_format = ''.join(pair_components)
            
            # First try to get price from executed_trades (most recent first)
            query = """
                SELECT 
                    COALESCE(average_fill_price, 
                            (SELECT target_price FROM trade_decisions WHERE id = decision_id)) as price
                FROM executed_trades
                WHERE trading_pair = %s AND transaction_time > NOW() - INTERVAL '1 day'
                ORDER BY transaction_time DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=[exchange_format])
            
            if not df.empty and df.iloc[0]['price'] is not None:
                return float(df.iloc[0]['price'])
                
            # If no recent executed trades, try trades table
            query = """
                SELECT price
                FROM trades
                WHERE trading_pair = %s AND timestamp > NOW() - INTERVAL '1 day'
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=[self.trading_pair])
            
            if not df.empty and df.iloc[0]['price'] is not None:
                return float(df.iloc[0]['price'])
                
            # If still no price, try to get from predictions (might have external price info)
            query = """
                SELECT external_price
                FROM tft_predictions
                WHERE trading_pair = %s AND prediction_timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY prediction_timestamp DESC
                LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn, params=[self.trading_pair])
            
            if not df.empty and 'external_price' in df.columns and df.iloc[0]['external_price'] is not None:
                return float(df.iloc[0]['external_price'])
                
            return None
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None
        finally:
            if conn:
                self.db_logger._return_connection(conn)

    async def fetch_recent_trades(self, limit=20):
        """Fetch recent trades from both trades and executed_trades tables"""
        if not self.db_logger:
            return []
        
        try:
            conn = self.db_logger._get_connection()
            if not conn:
                logger.error("Failed to get database connection")
                return []
            
            # Extract the trading pair components for different formats
            # WLD-USDT → WLDUSDT
            pair_components = self.trading_pair.split('-')
            exchange_format = ''.join(pair_components)
            
            logger.info(f"Fetching trades for both formats: '{self.trading_pair}' and '{exchange_format}'")
            
            # Query all trades for PnL calculation
            query = """
                WITH all_trades AS (
                    -- Get trades from trades table
                    SELECT 
                        id,
                        'trades' as source,
                        timestamp, 
                        trading_pair, 
                        direction, 
                        price, 
                        size, 
                        action,
                        status,
                        order_type,
                        pnl as recorded_pnl,
                        fees,
                        NULL as decision_text,
                        NULL as trade_decision_id,
                        NULL as target_price,
                        NULL as requested_qty
                    FROM trades
                    WHERE trading_pair = %s
                    
                    UNION ALL
                    
                    -- Get trades from executed_trades with trade_decisions
                    SELECT 
                        et.id,
                        'executed_trades' as source,
                        et.transaction_time as timestamp, 
                        et.trading_pair, 
                        et.side as direction, 
                        et.average_fill_price as price,
                        et.filled_quantity as size, 
                        CASE 
                            WHEN td.decision LIKE 'ENTER%%' THEN 'OPEN'
                            WHEN td.decision LIKE 'EXIT%%' THEN 'CLOSE'
                            ELSE 'UNKNOWN'
                        END as action,
                        et.status,
                        et.order_type,
                        NULL as recorded_pnl,
                        et.commission as fees,
                        td.decision as decision_text,
                        td.id as trade_decision_id,
                        td.target_price,
                        et.requested_quantity as requested_qty
                    FROM executed_trades et
                    JOIN trade_decisions td ON et.decision_id = td.id
                    WHERE et.trading_pair = %s
                )
                SELECT * FROM all_trades
                ORDER BY timestamp DESC
                LIMIT %s
            """
            
            logger.info(f"Executing trades query for {self.trading_pair}/{exchange_format}, limit {limit}")
            df = pd.read_sql_query(query, conn, params=[self.trading_pair, exchange_format, limit])
            logger.info(f"Found {len(df)} recent trades")
            
            # Calculate PnL by pairing trades
            if not df.empty:
                # Sort by timestamp ascending for proper pairing
                df = df.sort_values('timestamp', ascending=True)
                
                # Pair OPEN and CLOSE trades to calculate PnL
                open_trades = []
                trade_pairs = []
                
                for _, trade in df.iterrows():
                    trade_dict = trade.to_dict()
                    if trade['action'] == 'OPEN':
                        # Store open trade
                        open_trades.append(trade_dict)
                    elif trade['action'] == 'CLOSE':
                        # Try to match with the oldest open trade with opposite direction
                        matched = False
                        for i, open_trade in enumerate(open_trades):
                            # For a valid pair: BUY-OPEN should close with SELL-CLOSE and vice versa
                            if (open_trade['direction'] == 'BUY' and trade['direction'] == 'SELL') or \
                               (open_trade['direction'] == 'SELL' and trade['direction'] == 'BUY'):
                                # We found a match
                                # Calculate PnL
                                if open_trade['price'] and trade['price']:
                                    try:
                                        open_price = float(open_trade['price'])
                                        close_price = float(trade['price'])
                                        size = min(float(open_trade['size'] or 0), float(trade['size'] or 0))
                                        
                                        if open_trade['direction'] == 'BUY':  # Long position
                                            # For long: PnL = (sell_price - buy_price) * size
                                            calculated_pnl = (close_price - open_price) * size
                                        else:  # Short position
                                            # For short: PnL = (buy_price - sell_price) * size
                                            calculated_pnl = (open_price - close_price) * size
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Error calculating PnL: {e}")
                                        calculated_pnl = None
                                else:
                                    calculated_pnl = None
                                
                                # Use recorded PnL if available (from trades table), otherwise use calculated
                                pnl = 0.0
                                if trade['recorded_pnl'] is not None and not pd.isna(trade['recorded_pnl']):
                                    try:
                                        pnl = float(trade['recorded_pnl'])
                                    except (ValueError, TypeError):
                                        pass
                                elif calculated_pnl is not None:
                                    pnl = calculated_pnl
                                
                                # Handle NaN values
                                if pd.isna(pnl):
                                    logger.warning(f"Found NaN PnL for trade pair {open_trade['id']}-{trade['id']}, using 0")
                                    pnl = 0.0
                                
                                # Add to statistics
                                trade_pairs.append({
                                    'open_id': open_trade['id'],
                                    'close_id': trade['id'],
                                    'pnl': pnl
                                })
                                
                                # Remove the matched open trade
                                open_trades.pop(i)
                                matched = True
                                break
                        
                        if matched:
                            # Update trade with paired info
                            for i, t in enumerate(df.iloc):
                                if t['id'] == trade['id'] and t['source'] == trade['source']:
                                    for key, value in trade_dict.items():
                                        df.at[i, key] = value
                
                # Sort by timestamp in descending order for display
                df = df.sort_values('timestamp', ascending=False)
                most_recent = df.iloc[0]['timestamp']
                logger.info(f"Most recent trade timestamp: {most_recent}")
            
            # Convert to dict and return
            self.recent_trades = df.to_dict('records')
            return self.recent_trades
            
        except Exception as e:
            logger.error(f"Error fetching recent trades: {str(e)}")
            self.add_error(f"Failed to fetch trades: {str(e)}")
            return []
        finally:
            if conn:
                self.db_logger._return_connection(conn)

    async def fetch_recent_predictions(self, limit=10):
        """Fetch recent model predictions from the database"""
        if not self.db_logger:
            return []
        
        try:
            conn = self.db_logger._get_connection()
            if not conn:
                logger.error("Failed to get database connection")
                return []
            
            query = """
                SELECT 
                    prediction_timestamp, 
                    trading_pair, 
                    dir_prediction, 
                    down_prediction, 
                    ensemble_prediction
                FROM tft_predictions
                WHERE trading_pair = %s
                ORDER BY prediction_timestamp DESC
                LIMIT %s
            """
            
            df = pd.read_sql_query(query, conn, params=[self.trading_pair, limit])
            self.recent_predictions = df.to_dict('records')
            return self.recent_predictions
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {str(e)}")
            self.add_error(f"Failed to fetch predictions: {str(e)}")
            return []
        finally:
            if conn:
                self.db_logger._return_connection(conn)

    async def fetch_pnl_stats(self, trading_pair, period_days=7):
        """Fetch PnL stats for the given trading pair"""
        try:
            if not self.db_logger:
                logger.error("Database logger not initialized, can't fetch PnL stats")
                return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0}
            
            # Use standardized PnL tracking
            from utils.pnl_tracker import calculate_pnl_stats
            
            # Get a database connection
            conn = self.db_logger._get_connection()
            if not conn:
                logger.error("Failed to get database connection for PnL stats")
                return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0}
                
            try:
                # Calculate PnL stats using the standardized method
                stats = calculate_pnl_stats(conn)
                
                # Store the stats in the instance variable
                self.pnl_stats = {
                    "total_pnl": stats["total_pnl"],
                    "win_rate": stats["win_rate"],
                    "total_trades": stats["total_trades"],
                    "winning_trades": stats["win_count"],
                    "closed_pnl": stats["total_pnl"],
                    "losing_trades": stats["total_trades"] - stats["win_count"]
                }
                
                logger.info(f"Calculated PnL stats: {self.pnl_stats['total_trades']} trades, "
                         f"{self.pnl_stats['winning_trades']} winning, "
                         f"{self.pnl_stats['total_pnl']:.6f} total PnL")
                
                return self.pnl_stats
            except Exception as e:
                logger.error(f"Error fetching PnL stats: {str(e)}")
                return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0}
            finally:
                # Return connection to pool
                if conn:
                    self.db_logger._return_connection(conn)
        except Exception as e:
            logger.error(f"Error fetching PnL stats: {str(e)}")
            return {"total_pnl": 0.0, "win_rate": 0.0, "total_trades": 0, "winning_trades": 0}

    def check_bot_status(self):
        """Check if the bot process is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "realtime_tft_bot.py"], 
                capture_output=True, 
                text=True
            )
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Error checking bot status: {str(e)}")
            self.add_error(f"Failed to check bot status: {str(e)}")
            return False

    def add_error(self, message):
        """Add an error message to the error log"""
        self.error_messages.append({
            'timestamp': datetime.now(),
            'message': message
        })
        # Keep only the last 10 errors
        if len(self.error_messages) > 10:
            self.error_messages = self.error_messages[-10:]

    async def refresh_data(self):
        """Refresh all data from the database"""
        logger.info("Refreshing data from database...")
        
        await self.fetch_current_position()
        await self.fetch_recent_trades()
        await self.fetch_recent_predictions()
        await self.fetch_pnl_stats(self.trading_pair)
        
        self.last_updated = datetime.now()
        logger.info(f"Data refresh completed at {self.last_updated}")

    def render(self, stdscr):
        """Render the monitoring interface"""
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
        
        # Color constants
        GREEN = curses.color_pair(1)
        RED = curses.color_pair(2)
        YELLOW = curses.color_pair(3)
        CYAN = curses.color_pair(4)
        HEADER = curses.color_pair(5)
        
        # Initial screen setup
        stdscr.clear()
        stdscr.refresh()
        
        # Get screen dimensions
        max_y, max_x = stdscr.getmaxyx()
        
        # Add this line to store the refresh task
        refresh_task = None
        
        # Main loop
        while self.is_running:
            try:
                # Clear screen
                stdscr.clear()
                
                # Header
                header = f"TFT Bot Monitor - {self.trading_pair} - Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
                stdscr.addstr(0, 0, header.center(max_x), HEADER)
                
                # Bot status
                bot_running = self.check_bot_status()
                status_text = "Bot Status: RUNNING" if bot_running else "Bot Status: STOPPED"
                status_color = GREEN if bot_running else RED
                stdscr.addstr(1, 2, status_text, status_color | curses.A_BOLD)
                
                # Right-aligned instructions
                instructions = "Press 'q' to quit, 'r' to refresh"
                stdscr.addstr(1, max_x - len(instructions) - 2, instructions, YELLOW)
                
                # Position information section
                stdscr.addstr(3, 2, "Current Position:", curses.A_BOLD)
                if self.position_state:
                    position = self.position_state['position']
                    position_color = GREEN if position == 'LONG' else RED if position == 'SHORT' else YELLOW
                    
                    stdscr.addstr(4, 4, f"State: ", curses.A_BOLD)
                    stdscr.addstr(f"{position}", position_color | curses.A_BOLD)
                    
                    if position != 'NONE':
                        entry_price = self.position_state['entry_price']
                        size = self.position_state['position_size']
                        entry_time = self.position_state['entry_timestamp']
                        holding_periods = self.position_state['holding_periods']
                        
                        # Display current market price if available
                        if 'current_price' in self.position_state:
                            current_price = self.position_state['current_price']
                            stdscr.addstr(5, 4, f"Entry Price: {entry_price}")
                            stdscr.addstr(6, 4, f"Current Price: {current_price}")
                            stdscr.addstr(7, 4, f"Size: {size}")
                            
                            # Display unrealized PnL if available
                            if 'unrealized_pnl' in self.position_state:
                                unrealized_pnl = self.position_state['unrealized_pnl']
                                pnl_percent = self.position_state['pnl_percent']
                                pnl_color = GREEN if unrealized_pnl > 0 else RED
                                
                                stdscr.addstr(8, 4, f"Unrealized PnL: ")
                                stdscr.addstr(f"{unrealized_pnl:.6f} ({pnl_percent:.2f}%)", pnl_color | curses.A_BOLD)
                            
                            stdscr.addstr(9, 4, f"Entry Time: {entry_time}")
                            stdscr.addstr(10, 4, f"Holding Periods: {holding_periods}")
                        else:
                            stdscr.addstr(5, 4, f"Entry Price: {entry_price}")
                            stdscr.addstr(6, 4, f"Size: {size}")
                            stdscr.addstr(7, 4, f"Entry Time: {entry_time}")
                            stdscr.addstr(8, 4, f"Holding Periods: {holding_periods}")
                else:
                    stdscr.addstr(4, 4, "No position data available", YELLOW)
                
                # Recent trades section
                stdscr.addstr(12, 2, "Recent Trades:", curses.A_BOLD)
                if self.recent_trades:
                    # Header
                    stdscr.addstr(13, 4, "Time".ljust(19) + "Action".ljust(8) + "Direction".ljust(9) + "Price".ljust(10) + "Size".ljust(10) + "Status".ljust(8) + "PnL")
                    
                    # List trades
                    for i, trade in enumerate(self.recent_trades[:5]):  # Show only 5 most recent
                        row = 14 + i
                        if row >= max_y - 15:  # Prevent going out of bounds
                            break
                            
                        source = trade.get('source', 'unknown')
                        time_str = trade['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                        action = trade['action']
                        direction = trade['direction']
                        
                        # For executed trades with NEW status, use target_price instead
                        if source == 'executed_trades' and trade.get('status') == 'NEW':
                            price = trade.get('target_price')
                            size = trade.get('requested_qty')
                        else:
                            price = trade.get('price')
                            size = trade.get('size')
                        
                        price_str = f"{price:.4f}" if price else "N/A"
                        size_str = f"{size:.4f}" if size else "N/A"
                        status = trade.get('status', 'UNKNOWN')
                        
                        # Set colors based on source and status
                        source_color = CYAN if source == 'executed_trades' else YELLOW
                        action_color = GREEN if action == 'CLOSE' else YELLOW
                        status_color = GREEN if status == 'FILLED' else YELLOW if status == 'NEW' else RED
                        direction_color = GREEN if direction == 'BUY' else RED
                        
                        # Get PnL string
                        pnl_str = ""
                        pnl_color = curses.A_NORMAL
                        
                        if action == 'CLOSE':
                            # First check for calculated_pnl (from our pairing logic)
                            if 'calculated_pnl' in trade and trade['calculated_pnl'] is not None:
                                pnl = trade['calculated_pnl']
                                pnl_str = f"{pnl:.4f}"
                                pnl_color = GREEN if pnl > 0 else RED
                            # Then check recorded_pnl from database
                            elif trade.get('recorded_pnl') is not None:
                                pnl = trade['recorded_pnl']
                                pnl_str = f"{pnl:.4f}"
                                pnl_color = GREEN if pnl > 0 else RED
                            # For new orders with status='NEW', try to calculate from paired entry
                            elif source == 'executed_trades' and status == 'NEW':
                                # If we have matched open trade data
                                if 'open_price' in trade and trade['open_price'] and price:
                                    if trade['open_direction'] == 'BUY':  # Long position
                                        est_pnl = price - trade['open_price']
                                    else:  # Short position
                                        est_pnl = trade['open_price'] - price
                                    
                                    # Format with asterisk to indicate estimate
                                    pnl_str = f"{est_pnl:.4f}*"
                                    pnl_color = GREEN if est_pnl > 0 else RED
                        
                        # For executed trades, add a prefix to indicate source and status
                        prefix = ""
                        if source == 'executed_trades':
                            decision = trade.get('decision_text', '')
                            if decision:
                                prefix = f"[{decision[:4]}] "  # First 4 chars of decision type
                            
                            if status == 'NEW':
                                # Add an asterisk to prices from trade_decisions to indicate estimate
                                if price_str != "N/A":
                                    price_str = f"{price_str}*"
                        
                        stdscr.addstr(row, 4, time_str.ljust(19), source_color)
                        stdscr.addstr((prefix + action).ljust(8), action_color)
                        stdscr.addstr(direction.ljust(9), direction_color)
                        stdscr.addstr(price_str.ljust(10))
                        stdscr.addstr(size_str.ljust(10))
                        stdscr.addstr(status.ljust(8), status_color)
                        stdscr.addstr(pnl_str, pnl_color)
                else:
                    stdscr.addstr(13, 4, "No recent trades", YELLOW)
                
                # Recent predictions section
                stdscr.addstr(20, 2, "Recent Predictions:", curses.A_BOLD)
                if self.recent_predictions:
                    # Header
                    stdscr.addstr(21, 4, "Time".ljust(19) + "Dir Model".ljust(12) + "Down Model".ljust(12) + "Ensemble")
                    
                    # List predictions
                    for i, pred in enumerate(self.recent_predictions[:5]):  # Show only 5 most recent
                        row = 22 + i
                        if row >= max_y - 8:  # Prevent going out of bounds
                            break
                            
                        time_str = pred['prediction_timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                        dir_pred = f"{pred['dir_prediction']:.4f}"
                        down_pred = f"{pred['down_prediction']:.4f}"
                        ensemble = f"{pred['ensemble_prediction']:.4f}"
                        
                        dir_color = GREEN if float(dir_pred) > 0 else RED
                        down_color = GREEN if float(down_pred) > 0 else RED
                        ensemble_color = GREEN if float(ensemble) > 0 else RED
                        
                        stdscr.addstr(row, 4, time_str.ljust(19))
                        stdscr.addstr(dir_pred.ljust(12), dir_color)
                        stdscr.addstr(down_pred.ljust(12), down_color)
                        stdscr.addstr(ensemble, ensemble_color)
                else:
                    stdscr.addstr(21, 4, "No recent predictions", YELLOW)
                
                # PnL Statistics 
                stdscr.addstr(3, max_x // 2 + 2, "PnL Statistics (7 days):", curses.A_BOLD)
                if self.pnl_stats:
                    total_pnl = self.pnl_stats.get('total_pnl', 0)
                    pnl_color = GREEN if total_pnl > 0 else RED
                    
                    stdscr.addstr(4, max_x // 2 + 4, f"Total PnL: ")
                    stdscr.addstr(f"{total_pnl:.4f}", pnl_color | curses.A_BOLD)
                    
                    stdscr.addstr(5, max_x // 2 + 4, f"Total Trades: {self.pnl_stats.get('total_trades', 0)}")
                    stdscr.addstr(6, max_x // 2 + 4, f"Winning Trades: {self.pnl_stats.get('winning_trades', 0)}")
                    
                    win_rate = self.pnl_stats.get('win_rate', 0)
                    win_rate_color = GREEN if win_rate > 50 else YELLOW if win_rate > 30 else RED
                    stdscr.addstr(7, max_x // 2 + 4, f"Win Rate: ")
                    stdscr.addstr(f"{win_rate:.1f}%", win_rate_color)
                else:
                    stdscr.addstr(4, max_x // 2 + 4, "No PnL data available", YELLOW)
                
                # Error messages section
                stdscr.addstr(max_y - 8, 2, "Recent Errors:", curses.A_BOLD)
                if self.error_messages:
                    for i, error in enumerate(reversed(self.error_messages[-3:])):  # Show last 3 errors
                        row = max_y - 7 + i
                        if row >= max_y - 1:  # Prevent going out of bounds
                            break
                            
                        time_str = error['timestamp'].strftime("%H:%M:%S")
                        message = error['message']
                        display_msg = f"{time_str} - {message}"
                        
                        # Truncate if too long
                        if len(display_msg) > max_x - 6:
                            display_msg = display_msg[:max_x - 9] + "..."
                            
                        stdscr.addstr(row, 4, display_msg, RED)
                else:
                    stdscr.addstr(max_y - 7, 4, "No errors", GREEN)
                
                # Checklist progress (bottom line)
                stdscr.addstr(max_y - 2, 2, "See realtime_tft_bot/checklist.md for complete monitoring checklist", CYAN)
                
                # Footer
                footer = f"Refresh Interval: {self.refresh_interval}s"
                stdscr.addstr(max_y - 1, max_x - len(footer) - 2, footer)
                
                # Refresh the screen
                stdscr.refresh()
                
                # Handle key presses
                stdscr.nodelay(True)
                key = stdscr.getch()
                
                if key == ord('q'):
                    self.is_running = False
                    break
                elif key == ord('r'):
                    # Make manual refresh happen in a thread to avoid blocking UI
                    stdscr.addstr(1, max_x - 30, "Refreshing...", YELLOW)
                    stdscr.refresh()
                    
                    # Run refresh in a thread
                    import threading
                    def refresh_thread():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.refresh_data())
                        loop.close()
                    
                    t = threading.Thread(target=refresh_thread)
                    t.daemon = True
                    t.start()
                
                # Wait before next refresh
                time.sleep(0.1)
                
                # Auto-refresh after interval
                current_time = datetime.now()
                time_since_refresh = (current_time - self.last_updated).total_seconds()
                if time_since_refresh >= self.refresh_interval:
                    logger.info(f"Auto-refreshing data after {time_since_refresh:.1f}s")
                    
                    # Run refresh in a thread to avoid asyncio issues
                    import threading
                    def auto_refresh_thread():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self.refresh_data())
                            loop.close()
                        except Exception as e:
                            logger.error(f"Error during auto-refresh: {str(e)}")
                            self.add_error(f"Refresh failed: {str(e)}")
                    
                    t = threading.Thread(target=auto_refresh_thread)
                    t.daemon = True
                    t.start()
            
            except curses.error:
                # Handle curses errors (like window resizing)
                pass
            
            except Exception as e:
                logger.error(f"Render error: {str(e)}")
                self.add_error(f"UI error: {str(e)}")
                time.sleep(1)  # Prevent error spam

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TFT Bot Monitoring UI")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TFTBotMonitor()
    monitor.refresh_interval = args.refresh
    
    if not await monitor.initialize():
        print("Failed to initialize monitor. Exiting.")
        return 1
    
    try:
        # Initial data load
        await monitor.refresh_data()
        
        # Start the UI
        curses.wrapper(monitor.render)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main()) 