#!/usr/bin/env python3
"""
Check Position State

This script examines the position_state table to find actual trade prices and sizes.
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
logger = logging.getLogger("CheckPositionState")

async def check_position_state():
    """Check the structure and data in the position_state table"""
    
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
                # First get schema of position_state table
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'position_state'
                    ORDER BY ordinal_position
                """)
                
                columns = cursor.fetchall()
                print("\n--- Position State Table Schema ---")
                for col in columns:
                    print(f"{col[0]}: {col[1]}")
                
                # Get position state
                cursor.execute("""
                    SELECT * FROM position_state
                """)
                
                positions = cursor.fetchall()
                
                if not positions:
                    print("\nNo positions found in the database")
                    return 0
                
                # Print header based on column names
                column_names = [desc[0] for desc in cursor.description]
                print(f"\n--- Position State Data ---")
                print(f"{', '.join(column_names)}")
                print("-" * 120)
                
                # Print each position with proper formatting
                for position in positions:
                    formatted_values = []
                    for i, value in enumerate(position):
                        if isinstance(value, float):
                            formatted_values.append(f"{value:.8f}")
                        else:
                            formatted_values.append(str(value))
                    
                    print(f"{', '.join(formatted_values)}")
                
                # Check entries made around the time of an order
                # Get the timestamp of a recent executed trade
                cursor.execute("""
                    SELECT transaction_time 
                    FROM executed_trades 
                    ORDER BY transaction_time DESC 
                    LIMIT 1
                """)
                
                latest_trade_time = cursor.fetchone()[0]
                
                # Check for any database entries around that time
                if latest_trade_time:
                    print(f"\n--- Recent Order Execution Data ({latest_trade_time}) ---")
                    
                    # Check for model predictions around that time
                    cursor.execute("""
                        SELECT prediction_timestamp, ensemble_prediction
                        FROM tft_predictions
                        WHERE ABS(EXTRACT(EPOCH FROM (prediction_timestamp - %s))) < 120
                        ORDER BY prediction_timestamp
                    """, (latest_trade_time,))
                    
                    predictions = cursor.fetchall()
                    if predictions:
                        print(f"\nPredictions around trade time:")
                        for pred in predictions:
                            print(f"  {pred[0]}: {pred[1]}")
                    
                    # Check for trade decisions around that time
                    cursor.execute("""
                        SELECT decision_timestamp, decision, prediction_value
                        FROM trade_decisions
                        WHERE ABS(EXTRACT(EPOCH FROM (decision_timestamp - %s))) < 120
                        ORDER BY decision_timestamp
                    """, (latest_trade_time,))
                    
                    decisions = cursor.fetchall()
                    if decisions:
                        print(f"\nTrade decisions around trade time:")
                        for dec in decisions:
                            print(f"  {dec[0]}: {dec[1]} ({dec[2]})")
        
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

if __name__ == "__main__":
    asyncio.run(check_position_state()) 