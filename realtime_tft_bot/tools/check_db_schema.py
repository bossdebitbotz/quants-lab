#!/usr/bin/env python3
"""
Database Schema Verification Tool

This script checks if all required database tables are present
and have the expected structure.
"""

import os
import sys
import asyncio
import logging
import argparse

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

logger = logging.getLogger("DBSchemaDiagnostic")

# Tables that should exist in the database
EXPECTED_TABLES = [
    'tft_predictions',
    'trade_decisions',
    'executed_trades',
    'position_state',
    'trades',
    'order_events',
    'performance_metrics'
]

async def check_db_schema():
    """Check if all required tables exist in the database"""
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
            missing_tables = []
            with conn.cursor() as cursor:
                # Check for each expected table
                for table in EXPECTED_TABLES:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        );
                    """, (table,))
                    
                    result = cursor.fetchone()[0]
                    if not result:
                        missing_tables.append(table)
                        logger.error(f"Missing table: {table}")
                    else:
                        logger.info(f"Table exists: {table}")
                        
                        # Check column count for more detailed verification
                        cursor.execute("""
                            SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_schema = 'public' AND table_name = %s;
                        """, (table,))
                        
                        column_count = cursor.fetchone()[0]
                        logger.info(f"  Table {table} has {column_count} columns")
                        
            if missing_tables:
                logger.error(f"Missing tables: {', '.join(missing_tables)}")
                logger.info("Consider running a fresh database initialization")
                return 1
            else:
                logger.info("All expected tables exist in the database")
                
            # Check specifically for position_state schema
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns 
                    WHERE table_name = 'position_state'
                """)
                
                columns = [row[0] for row in cursor.fetchall()]
                logger.info(f"Position state table columns: {', '.join(columns)}")
                
                # Check for required columns
                required_columns = [
                    'trading_pair', 'position', 'entry_price', 
                    'position_size', 'entry_timestamp', 'holding_periods',
                    'last_update_timestamp'
                ]
                
                missing_columns = [col for col in required_columns if col not in columns]
                if missing_columns:
                    logger.error(f"Missing columns in position_state: {', '.join(missing_columns)}")
                    return 1
                else:
                    logger.info("Position state table has all required columns")
            
            return 0
                
        finally:
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Database Schema Verification Tool")
    args = parser.parse_args()
    
    return asyncio.run(check_db_schema())

if __name__ == "__main__":
    sys.exit(main()) 