#!/usr/bin/env python3
"""
View All Trades Script

This script combines and displays trades from both the 'trades' and 'executed_trades' tables
to provide a complete trading history.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timezone

# Add the realtime_tft_bot directory to the path for imports
sys.path.insert(0, os.path.abspath('realtime_tft_bot'))

from utils.db_logger import DBLogger
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViewAllTrades")

def make_naive(dt):
    """Convert a datetime to naive (no timezone info)"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC then remove timezone info
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt

async def get_all_trades():
    """Fetch and display trades from both tables to show complete history"""
    
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
                # Get old trades from the 'trades' table
                cursor.execute("""
                    SELECT 
                        'trades' as source,
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
                """)
                
                old_trades = cursor.fetchall()
                
                # Get recent trades from the 'executed_trades' table, joining with trade_decisions for additional context
                cursor.execute("""
                    SELECT 
                        'executed_trades' as source,
                        et.id,
                        et.transaction_time as timestamp,
                        et.trading_pair,
                        et.side as direction,
                        et.average_fill_price as price,
                        et.filled_quantity as size,
                        CASE 
                            WHEN td.decision LIKE 'ENTER%' THEN 'OPEN'
                            WHEN td.decision LIKE 'EXIT%' THEN 'CLOSE'
                            ELSE 'UNKNOWN'
                        END as action,
                        et.order_type,
                        et.status,
                        NULL as pnl,
                        et.commission as fees
                    FROM executed_trades et
                    LEFT JOIN trade_decisions td ON et.decision_id = td.id
                    ORDER BY et.transaction_time DESC
                """)
                
                new_trades = cursor.fetchall()
                
                # Process timestamps to make all datetimes naive
                old_trades = [list(t[:2]) + [make_naive(t[2])] + list(t[3:]) for t in old_trades]
                new_trades = [list(t[:2]) + [make_naive(t[2])] + list(t[3:]) for t in new_trades]
                
                # Combine both result sets
                all_trades = new_trades + old_trades
                
                if not all_trades:
                    print("No trades found in either table")
                    return 0
                
                # Sort by timestamp
                all_trades.sort(key=lambda x: x[2] if x[2] is not None else datetime.min, reverse=True)
                
                # Calculate how many trades to show
                # First 20 plus all of today's trades
                today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                todays_trades = [t for t in all_trades if t[2] and t[2].date() == today.date()]
                
                # Show today's trades + enough older trades to make at least 20
                trades_to_show = all_trades[:max(20, len(todays_trades))]
                
                # Print header
                print(f"\n{'Source':<15} {'ID':<5} {'Timestamp':<25} {'Pair':<10} {'Direction':<10} {'Price':<10} {'Size':<10} {'Action':<10} {'Status':<10} {'PnL':<10} {'Fees':<10}")
                print("-" * 140)
                
                # Print each trade
                for trade in trades_to_show:
                    source, id, timestamp, trading_pair, direction, price, size, action, order_type, status, pnl, fees = trade
                    
                    # Format display values
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'N/A'
                    pair_str = trading_pair if trading_pair else 'N/A'
                    dir_str = direction if direction else 'N/A'
                    price_str = f"{float(price):.6f}" if price else 'N/A'
                    size_str = f"{float(size):.6f}" if size else 'N/A'
                    action_str = action if action else 'N/A'
                    status_str = status if status else 'N/A'
                    pnl_str = f"{float(pnl):.6f}" if pnl else 'N/A'
                    fee_str = f"{float(fees):.6f}" if fees else 'N/A'
                    
                    print(f"{source:<15} {id:<5} {time_str:<25} {pair_str:<10} {dir_str:<10} {price_str:<10} {size_str:<10} {action_str:<10} {status_str:<10} {pnl_str:<10} {fee_str:<10}")
                
                # Print summary
                print("\nTotal trades found:", len(all_trades))
                print("Today's trades:", len(todays_trades))
                
                # Get count by source
                trades_count = len([t for t in all_trades if t[0] == 'trades'])
                executed_trades_count = len([t for t in all_trades if t[0] == 'executed_trades'])
                print(f"Trades in 'trades' table: {trades_count}")
                print(f"Trades in 'executed_trades' table: {executed_trades_count}")
                
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    try:
        asyncio.run(get_all_trades())
    except KeyboardInterrupt:
        print("\nQuery cancelled by user")
    except Exception as e:
        print(f"Error querying trades: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 