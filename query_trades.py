#!/usr/bin/env python3
"""
Query Trades Script

This script fetches the last 20 trades from the database and displays them.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add the realtime_tft_bot directory to the path for imports
sys.path.insert(0, os.path.abspath('realtime_tft_bot'))

from utils.db_logger import DBLogger
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QueryTrades")

async def get_recent_trades():
    """Fetch and display the last 20 trades from the database"""
    
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        # Get database connection
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
            
        try:
            with conn.cursor() as cursor:
                # Query the last 20 trades
                cursor.execute("""
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
                        pnl, 
                        fees
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT 20
                """)
                
                rows = cursor.fetchall()
                
                if not rows:
                    print("No trades found in the database")
                    return 0
                
                # Print header
                print(f"\n{'ID':<5} {'Timestamp':<25} {'Pair':<10} {'Direction':<10} {'Price':<10} {'Size':<10} {'Action':<10} {'PnL':<10} {'Fees':<10}")
                print("-" * 120)
                
                # Print each trade
                for row in rows:
                    id, timestamp, trading_pair, direction, price, size, action, order_type, status, pnl, fees = row
                    print(f"{id:<5} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<25} {trading_pair:<10} {direction:<10} {price:<10.6f} {size:<10.6f} {action:<10} {pnl if pnl else 0.0:<10.6f} {fees:<10.6f}")
                
                # Print summary
                print("\nTotal trades:", len(rows))
                
                # Print PnL summary for CLOSE trades only
                close_trades = [row for row in rows if row[6] == 'CLOSE']
                if close_trades:
                    total_pnl = sum(row[9] for row in close_trades if row[9])
                    total_fees = sum(row[10] for row in close_trades if row[10])
                    print(f"Total PnL (from displayed CLOSE trades): {total_pnl:.6f}")
                    print(f"Total fees (from displayed CLOSE trades): {total_fees:.6f}")
                    print(f"Net PnL (from displayed CLOSE trades): {total_pnl - total_fees:.6f}")
                
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    try:
        asyncio.run(get_recent_trades())
    except KeyboardInterrupt:
        print("\nQuery cancelled by user")
    except Exception as e:
        print(f"Error querying trades: {str(e)}")

if __name__ == "__main__":
    main() 