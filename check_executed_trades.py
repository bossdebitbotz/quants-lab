#!/usr/bin/env python3
"""
Check Executed Trades Script

This script fetches the executed_trades table to see recent trading activity.
"""

import os
import sys
import asyncio
import logging

# Add the realtime_tft_bot directory to the path for imports
sys.path.insert(0, os.path.abspath('realtime_tft_bot'))

from utils.db_logger import DBLogger
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckExecutedTrades")

async def get_executed_trades():
    """Fetch and display the most recent executed trades"""
    
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
                # First get schema of executed_trades to understand the structure
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'executed_trades'
                    ORDER BY ordinal_position
                """)
                
                columns = cursor.fetchall()
                print("\n--- Executed Trades Table Schema ---")
                for col in columns:
                    print(f"{col[0]}: {col[1]}")
                
                # Check the most recent 20 executed trades
                cursor.execute("""
                    SELECT et.id, et.decision_id, et.exchange_order_id, et.trading_pair, 
                           et.side, et.order_type, et.status, et.requested_quantity, 
                           et.filled_quantity, et.average_fill_price, et.transaction_time,
                           td.decision, td.prediction_value
                    FROM executed_trades et
                    LEFT JOIN trade_decisions td ON et.decision_id = td.id
                    ORDER BY et.transaction_time DESC
                    LIMIT 20
                """)
                
                trades = cursor.fetchall()
                
                if not trades:
                    print("\nNo executed trades found in the database")
                    return 0
                
                # Print header
                print(f"\n--- Last 20 Executed Trades ---")
                print(f"{'ID':<4} {'Time':<25} {'Pair':<10} {'Side':<6} {'Status':<10} {'Price':<10} {'Quantity':<10} {'Decision':<15}")
                print("-" * 120)
                
                # Print each trade
                for trade in trades:
                    id, decision_id, order_id, pair, side, order_type, status, req_qty, filled_qty, price, time, decision, pred_value = trade
                    
                    # Format for display
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S') if time else 'N/A'
                    price_str = f"{price:.6f}" if price else 'N/A'
                    qty_str = f"{filled_qty:.6f}" if filled_qty else 'N/A'
                    decision_str = decision if decision else 'N/A'
                    
                    print(f"{id:<4} {time_str:<25} {pair:<10} {side:<6} {status:<10} {price_str:<10} {qty_str:<10} {decision_str:<15}")
                
                # Also check the trade_decisions table to understand the relationship
                print("\n--- Last 5 Trade Decisions ---")
                cursor.execute("""
                    SELECT id, decision_timestamp, trading_pair, decision, prediction_value, reason
                    FROM trade_decisions
                    ORDER BY decision_timestamp DESC
                    LIMIT 5
                """)
                
                decisions = cursor.fetchall()
                if decisions:
                    print(f"{'ID':<4} {'Time':<25} {'Decision':<15} {'Prediction':<10} {'Reason':<30}")
                    print("-" * 90)
                    for decision in decisions:
                        id, time, pair, decision, pred, reason = decision
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S') if time else 'N/A'
                        pred_str = f"{pred:.6f}" if pred else 'N/A'
                        reason_str = reason if reason else 'N/A'
                        print(f"{id:<4} {time_str:<25} {decision:<15} {pred_str:<10} {reason_str:<30}")
                else:
                    print("No trade decisions found")
                
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    try:
        asyncio.run(get_executed_trades())
    except KeyboardInterrupt:
        print("\nQuery cancelled by user")
    except Exception as e:
        print(f"Error querying executed trades: {str(e)}")

if __name__ == "__main__":
    main() 