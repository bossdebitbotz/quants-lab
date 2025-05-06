#!/usr/bin/env python3
"""
Check Trade Decisions Table

This script examines the structure and content of the trade_decisions table
to help diagnose issues with PnL tracking.
"""

import os
import sys
import asyncio
import logging

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

logger = logging.getLogger("CheckTradeDecisions")

async def check_trade_decisions():
    """Check the structure and content of the trade_decisions table"""
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
                cursor.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'trade_decisions'")
                columns = cursor.fetchall()
                
                print("\nTrade Decisions Table Structure:")
                print("=" * 70)
                print(f"{'Column Name':<30} {'Data Type':<15} {'Nullable':<10}")
                print("-" * 70)
                for col in columns:
                    print(f"{col[0]:<30} {col[1]:<15} {col[2]:<10}")
                
                # Get latest decisions
                column_names = [col[0] for col in columns]
                time_column = None
                
                # Find the timestamp column
                time_candidates = ['timestamp', 'created_at', 'decision_time']
                for candidate in time_candidates:
                    if candidate in column_names:
                        time_column = candidate
                        break
                
                if not time_column:
                    print("\nCould not find a timestamp column. Using default columns.")
                    query = """
                        SELECT 
                            id, 
                            trading_pair, 
                            decision,
                            target_price
                        FROM trade_decisions 
                        ORDER BY id DESC 
                        LIMIT 10
                    """
                else:
                    query = f"""
                        SELECT 
                            id, 
                            trading_pair, 
                            decision,
                            target_price,
                            {time_column}
                        FROM trade_decisions 
                        ORDER BY {time_column} DESC 
                        LIMIT 10
                    """
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                headers = ['ID', 'Pair', 'Decision', 'Target Price']
                if time_column:
                    headers.append('Time')
                
                print("\nLatest Trade Decisions:")
                print("=" * 100)
                print(' '.join(f"{h:<20}" for h in headers))
                print("-" * 100)
                
                for row in rows:
                    formatted_row = [str(val)[:19] if i == len(row)-1 and time_column else str(val) for i, val in enumerate(row)]
                    print(' '.join(f"{val:<20}" for val in formatted_row))
                
                # Count decisions by type
                cursor.execute("""
                    SELECT 
                        decision,
                        COUNT(*) as count
                    FROM trade_decisions
                    GROUP BY decision
                    ORDER BY count DESC
                """)
                decision_counts = cursor.fetchall()
                
                print("\nDecision Types:")
                print("=" * 40)
                print(f"{'Decision':<20} {'Count':<10}")
                print("-" * 40)
                for decision_type, count in decision_counts:
                    print(f"{decision_type:<20} {count:<10}")
                
                # Check relationship with executed_trades
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT td.id) as total_decisions,
                        COUNT(DISTINCT et.decision_id) as decisions_with_trades,
                        COUNT(DISTINCT td.id) - COUNT(DISTINCT et.decision_id) as decisions_without_trades
                    FROM trade_decisions td
                    LEFT JOIN executed_trades et ON td.id = et.decision_id
                """)
                relationship = cursor.fetchone()
                
                print("\nRelationship with Executed Trades:")
                print(f"Total decisions: {relationship[0]}")
                print(f"Decisions with executed trades: {relationship[1]} ({relationship[1]/relationship[0]*100 if relationship[0] > 0 else 0:.2f}%)")
                print(f"Decisions without executed trades: {relationship[2]} ({relationship[2]/relationship[0]*100 if relationship[0] > 0 else 0:.2f}%)")
                
            return 0
        finally:
            db_logger._return_connection(conn)
    finally:
        # Close database connection
        await db_logger.close()

async def main():
    return await check_trade_decisions()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 