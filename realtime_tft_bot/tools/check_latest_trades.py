#!/usr/bin/env python3
"""
Latest Trades Diagnostic Tool

This script directly queries the database for the most recent trades 
to help diagnose issues with the TFT Bot Monitor.
"""

import os
import sys
import asyncio
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("LatestTradesDiagnostic")

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

try:
    from utils.db_logger import DBLogger
    from config import load_config
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    print(f"Error: {str(e)}")
    print("Ensure you've run setup_monitor.sh to install dependencies")
    sys.exit(1)

async def get_latest_trades(limit=10, minutes=60, trading_pair=None):
    """
    Fetch the latest trades from the database
    
    Args:
        limit: Maximum number of trades to fetch
        minutes: Only fetch trades from the last N minutes
        trading_pair: Filter by specific trading pair
    """
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        # If trading pair not specified, use from config
        if not trading_pair:
            trading_pair = config.get('TRADING_PAIR')
        
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
            
        try:
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(minutes=minutes)
            
            # Build query
            query = """
                SELECT 
                    id,
                    timestamp, 
                    trading_pair, 
                    direction, 
                    price, 
                    size, 
                    action, 
                    order_type,
                    status,
                    pnl
                FROM trades
                WHERE timestamp > %s
            """
            params = [time_threshold]
            
            if trading_pair:
                query += " AND trading_pair = %s"
                params.append(trading_pair)
                
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            # Execute query
            logger.info(f"Querying for trades in the last {minutes} minutes for {trading_pair}")
            df = pd.read_sql_query(query, conn, params=params)
            
            # Display results
            if df.empty:
                print(f"No trades found in the last {minutes} minutes for {trading_pair}")
            else:
                print(f"Found {len(df)} trades in the last {minutes} minutes for {trading_pair}:")
                print("\nMost recent trades (newest first):")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200)
                print(df)
                
                # Print specific information about the most recent trade
                latest = df.iloc[0]
                print("\nMost recent trade details:")
                print(f"ID: {latest['id']}")
                print(f"Timestamp: {latest['timestamp']} ({(datetime.now() - latest['timestamp']).total_seconds():.1f} seconds ago)")
                print(f"Pair: {latest['trading_pair']}")
                print(f"Action: {latest['action']}")
                print(f"Direction: {latest['direction']}")
                print(f"Price: {latest['price']}")
                print(f"Size: {latest['size']}")
                if latest['action'] == 'CLOSE':
                    print(f"PnL: {latest['pnl']}")
                
                # Check if trade is very recent (less than 30 seconds ago)
                if (datetime.now() - latest['timestamp']).total_seconds() < 30:
                    print("\nNOTE: A very recent trade was found. If it's not showing in the monitor, try:")
                    print("1. Manually refreshing the monitor with 'r' key")
                    print("2. Checking the bot_monitor.log file for errors")
                    print("3. Restarting the monitor")
                
        finally:
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Latest Trades Diagnostic Tool")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of trades to display")
    parser.add_argument("--minutes", type=int, default=60, help="Only fetch trades from the last N minutes")
    parser.add_argument("--pair", help="Trading pair to filter by (default: from config)")
    
    args = parser.parse_args()
    
    asyncio.run(get_latest_trades(args.limit, args.minutes, args.pair))

if __name__ == "__main__":
    main() 