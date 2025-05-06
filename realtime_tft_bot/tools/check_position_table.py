#!/usr/bin/env python3
"""
Position State Table Diagnostic Tool

This script directly queries the position_state table to understand what's
happening with the current trading position.
"""

import os
import sys
import asyncio
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("PositionStateDiagnostic")

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

async def check_position_state(trading_pair=None):
    """
    Fetch the current position state and history
    
    Args:
        trading_pair: Trading pair to filter by
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
        
        # Get current position state using the existing function
        position_state = await db_logger.get_position_state(trading_pair)
        
        print(f"\nCurrent position state for {trading_pair}:")
        if position_state:
            print(f"  Position: {position_state['position']}")
            print(f"  Entry Price: {position_state['entry_price']}")
            print(f"  Position Size: {position_state['position_size']}")
            print(f"  Entry Time: {position_state['entry_timestamp']}")
            print(f"  Holding Periods: {position_state['holding_periods']}")
            print(f"  Last Update: {position_state.get('last_update_timestamp')}")
            
            # Calculate how long position has been open
            if position_state['position'] != 'NONE' and position_state['entry_timestamp']:
                # Handle timezone-aware datetimes
                now = datetime.now(timezone.utc)
                time_open = now - position_state['entry_timestamp']
                print(f"  Time Open: {time_open}")
        else:
            print("  No position state found")
            
        # Now get the raw position_state table contents
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
            
        try:
            # Query position state history
            query = """
                SELECT *
                FROM position_state
                WHERE trading_pair = %s
                ORDER BY last_update_timestamp DESC
                LIMIT 10
            """
            
            df = pd.read_sql_query(query, conn, params=[trading_pair])
            
            if df.empty:
                print(f"\nNo position state history found for {trading_pair}")
            else:
                print(f"\nPosition state history for {trading_pair} (last 10 updates):")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 200)
                print(df)
            
            # Now check for any trades in the last day
            trades_query = """
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
                WHERE trading_pair = %s
                  AND timestamp > NOW() - INTERVAL '1 day'
                ORDER BY timestamp DESC
                LIMIT 10
            """
            
            trades_df = pd.read_sql_query(trades_query, conn, params=[trading_pair])
            
            if trades_df.empty:
                print(f"\nNo trades found in the last day for {trading_pair}")
            else:
                print(f"\nMost recent trades for {trading_pair} (last day):")
                print(trades_df)
                
            # Check for any trade decisions
            decisions_query = """
                SELECT 
                    timestamp,
                    trading_pair,
                    current_price,
                    prediction_value,
                    decision,
                    reason
                FROM trade_decisions
                WHERE trading_pair = %s
                  AND timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 10
            """
            
            decisions_df = pd.read_sql_query(decisions_query, conn, params=[trading_pair])
            
            if decisions_df.empty:
                print(f"\nNo trade decisions found in the last hour for {trading_pair}")
            else:
                print(f"\nMost recent trade decisions for {trading_pair} (last hour):")
                print(decisions_df)
                
        finally:
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Position State Diagnostic Tool")
    parser.add_argument("--pair", help="Trading pair to filter by (default: from config)")
    
    args = parser.parse_args()
    
    asyncio.run(check_position_state(args.pair))

if __name__ == "__main__":
    main() 