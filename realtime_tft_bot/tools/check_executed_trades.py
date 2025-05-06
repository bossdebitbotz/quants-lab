#!/usr/bin/env python3
"""
Check Executed Trades Table

This script examines the structure and content of the executed_trades table
to help diagnose issues with PnL tracking.
"""

import os
import sys
import asyncio
import logging
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db_logger import DBLogger
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("CheckExecutedTrades")

async def check_executed_trades():
    """Check the structure and content of the executed_trades table"""
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
            
        try:
            # Get table structure
            with conn.cursor() as cursor:
                cursor.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'executed_trades'")
                columns = cursor.fetchall()
                
                print("\nExecuted Trades Table Structure:")
                print("=" * 70)
                print(f"{'Column Name':<30} {'Data Type':<15} {'Nullable':<10}")
                print("-" * 70)
                for col in columns:
                    print(f"{col[0]:<30} {col[1]:<15} {col[2]:<10}")
                
                # Get sample trade data
                cursor.execute("SELECT id, decision_id, transaction_time, trading_pair, side, order_type, average_fill_price, filled_quantity, status FROM executed_trades LIMIT 5")
                trades = cursor.fetchall()
                
                if trades:
                    print("\nSample Executed Trades:")
                    for trade in trades:
                        trade_id, decision_id, timestamp, trading_pair, side, order_type, fill_price, quantity, status = trade
                        print("\n" + "=" * 70)
                        print(f"ID: {trade_id}, Decision ID: {decision_id}")
                        print(f"Timestamp: {timestamp}")
                        print(f"Trading Pair: {trading_pair}")
                        print(f"Side: {side}")
                        print(f"Order Type: {order_type}")
                        print(f"Fill Price: {fill_price}")
                        print(f"Quantity: {quantity}")
                        print(f"Status: {status}")
                else:
                    print("No executed trades found in the database")
                
                # Count executed trades
                cursor.execute("SELECT COUNT(*) FROM executed_trades")
                count = cursor.fetchone()[0]
                print(f"\nTotal executed trades: {count}")
                
                # Check relationship with trade_decisions
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT et.id) as total_executed,
                        COUNT(DISTINCT et.decision_id) as unique_decisions,
                        COUNT(DISTINCT td.id) as total_decisions
                    FROM executed_trades et
                    LEFT JOIN trade_decisions td ON et.decision_id = td.id
                """)
                relationship = cursor.fetchone()
                
                print("\nRelationship with Trade Decisions:")
                print(f"Total executed trades: {relationship[0]}")
                print(f"Unique decisions executed: {relationship[1]}")
                print(f"Total decisions: {relationship[2]}")
                print(f"Execution rate: {relationship[1]/relationship[2]*100 if relationship[2] > 0 else 0:.2f}%")
                
            return 0
        finally:
            db_logger._return_connection(conn)
    finally:
        # Close database connection
        await db_logger.close()

async def main():
    return await check_executed_trades()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 