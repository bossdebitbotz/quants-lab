#!/usr/bin/env python3
"""
Database Logger Diagnostic Tool

This script examines the DBLogger module and database structure
to understand why trades might not be recorded while positions are updated.
"""

import os
import sys
import asyncio
import logging
import inspect
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("DBLoggerDiagnostic")

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

try:
    from utils.db_logger import DBLogger
    from config import load_config
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    print(f"Error: {str(e)}")
    print("Ensure you've run setup_monitor.sh to install dependencies")
    sys.exit(1)

async def examine_db_logger():
    """Examine the DBLogger module and database structure"""
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1

    try:
        # 1. Examine the DBLogger class - print its methods
        print("\n--- DBLogger Methods ---")
        for name, method in inspect.getmembers(db_logger, predicate=inspect.ismethod):
            if not name.startswith('_'):  # Skip private methods
                print(f"Method: {name}")
                # Get the method signature
                signature = inspect.signature(method)
                print(f"  Signature: {signature}")

        # 2. Check database tables 
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
            
        try:
            # Get a list of all tables
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cursor.fetchall()
            
            print("\n--- Database Tables ---")
            for table in tables:
                table_name = table[0]
                print(f"Table: {table_name}")
                
                # Get table structure (columns)
                cursor.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                """)
                columns = cursor.fetchall()
                for column in columns:
                    print(f"  {column[0]}: {column[1]}")
            
            # 3. Check if log_trade method is called during trade execution
            print("\n--- Examining log_trade Functionality ---")
            # Look for methods that call log_trade
            print("Logic for logging trades should be in methods that call db_logger.log_trade()")
            print("Methods to check:")
            print("- execution_handler.py > execute_order() or handle_execution")
            print("- realtime_tft_bot.py > any methods that execute trades")
            print("- decision_engine.py > methods that implement trading decisions")
            
            # 4. Check trades table specifically
            print("\n--- Trades Table Content Check ---")
            try:
                cursor.execute("SELECT COUNT(*) FROM trades")
                count = cursor.fetchone()[0]
                print(f"Total trades in database: {count}")
                
                if count > 0:
                    cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 1")
                    latest = cursor.fetchone()
                    desc = cursor.description
                    print("Most recent trade:")
                    for i, col in enumerate(desc):
                        print(f"  {col[0]}: {latest[i]}")
                    
                    # Show a few more recent trades
                    print("\nRecent trades:")
                    cursor.execute("""
                        SELECT id, timestamp, trading_pair, direction, price, size, action, pnl
                        FROM trades 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    """)
                    recent_trades = cursor.fetchall()
                    for trade in recent_trades:
                        print(f"ID: {trade[0]}, Time: {trade[1]}, Pair: {trade[2]}, Action: {trade[6]}, PnL: {trade[7]}")
                else:
                    print("No trades found in the database")
            except Exception as e:
                print(f"Error accessing trades table: {str(e)}")
                
            # 5. Look for any errors related to trade logging
            print("\n--- Check for Common Issues ---")
            print("1. The code path that calls log_trade might not be reached")
            print("2. There could be exception handling that swallows errors during trade logging")
            print("3. The trades table might have constraints preventing insertions")
            print("4. Transactions might be rolling back due to errors elsewhere")
            print("5. The connection could be timing out when trying to log trades")
            
            # 6. Check for position_state updates
            print("\n--- Position State Updates ---")
            try:
                cursor.execute("""
                    SELECT trading_pair, position, entry_price, position_size, entry_timestamp, last_update_timestamp
                    FROM position_state
                    ORDER BY last_update_timestamp DESC
                    LIMIT 5
                """)
                positions = cursor.fetchall()
                if positions:
                    for pos in positions:
                        print(f"Pair: {pos[0]}, Position: {pos[1]}, Price: {pos[2]}, Size: {pos[3]}")
                        print(f"  Entry: {pos[4]}, Updated: {pos[5]}")
                else:
                    print("No position state records found")
            except Exception as e:
                print(f"Error accessing position_state table: {str(e)}")
            
        finally:
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    asyncio.run(examine_db_logger())

if __name__ == "__main__":
    main() 