#!/usr/bin/env python3
"""
Check Trade Decisions

This script examines the trade_decisions table to find price information.
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
logger = logging.getLogger("CheckTradeDecisions")

async def check_trade_decisions():
    """Check the trade_decisions table for price information"""
    
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
                # First get schema of trade_decisions table
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'trade_decisions'
                    ORDER BY ordinal_position
                """)
                
                columns = cursor.fetchall()
                print("\n--- Trade Decisions Table Schema ---")
                for col in columns:
                    print(f"{col[0]}: {col[1]}")
                
                # Get recent trade decisions
                cursor.execute("""
                    SELECT id, decision_timestamp, trading_pair, 
                           decision, target_price, prediction_value, reason
                    FROM trade_decisions
                    ORDER BY decision_timestamp DESC
                    LIMIT 10
                """)
                
                decisions = cursor.fetchall()
                
                if not decisions:
                    print("\nNo trade decisions found in the database")
                    return 0
                
                # Print header
                print(f"\n--- Recent Trade Decisions ---")
                print(f"{'ID':<4} {'Time':<25} {'Pair':<10} {'Decision':<15} {'Target Price':<12} {'Pred Value':<12} {'Reason'}")
                print("-" * 120)
                
                # Print each decision with proper formatting
                for decision in decisions:
                    id, time, pair, dec, price, pred, reason = decision
                    
                    # Format for display
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S') if time else 'N/A'
                    price_str = f"{price:.8f}" if price is not None else 'N/A'
                    pred_str = f"{pred:.8f}" if pred is not None else 'N/A'
                    reason_str = reason[:40] + '...' if reason and len(reason) > 40 else reason
                    
                    print(f"{id:<4} {time_str:<25} {pair:<10} {dec:<15} {price_str:<12} {pred_str:<12} {reason_str}")
                
                # Check the price accuracy by comparing to API/executed price
                print("\n--- Price Accuracy Check ---")
                cursor.execute("""
                    SELECT td.id, td.decision_timestamp, td.decision, td.target_price,
                           et.average_fill_price, et.requested_quantity
                    FROM trade_decisions td
                    JOIN executed_trades et ON td.id = et.decision_id
                    ORDER BY td.decision_timestamp DESC
                    LIMIT 10
                """)
                
                comparisons = cursor.fetchall()
                
                if comparisons:
                    print(f"{'ID':<4} {'Time':<25} {'Decision':<15} {'Target Price':<12} {'Actual Price':<12} {'Quantity'}")
                    print("-" * 100)
                    for comp in comparisons:
                        id, time, dec, target, actual, qty = comp
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S') if time else 'N/A'
                        target_str = f"{target:.8f}" if target is not None else 'N/A'
                        actual_str = f"{actual:.8f}" if actual is not None else 'N/A'
                        qty_str = f"{qty:.8f}" if qty is not None else 'N/A'
                        
                        print(f"{id:<4} {time_str:<25} {dec:<15} {target_str:<12} {actual_str:<12} {qty_str}")
                else:
                    print("No price comparisons available")
                
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

if __name__ == "__main__":
    asyncio.run(check_trade_decisions()) 