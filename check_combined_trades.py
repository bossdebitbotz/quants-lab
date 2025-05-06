#!/usr/bin/env python3
"""
Check Combined Trades Query

This script tests the SQL query used to combine trades from both tables
and diagnoses why executed trades aren't showing up in the monitor.
"""

import os
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime

# Add the realtime_tft_bot directory to the path for imports
sys.path.insert(0, os.path.abspath('realtime_tft_bot'))

from utils.db_logger import DBLogger
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckCombinedTrades")

async def test_combined_query():
    """Test the combined query directly to diagnose issues"""
    
    # Load configuration
    config = load_config()
    trading_pair = config.get('TRADING_PAIR', 'WLD-USDT')
    
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
        
        # Test the trades table structure
        logger.info("Checking trades table structure...")
        cursor = conn.cursor()
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'trades'")
        trades_columns = cursor.fetchall()
        logger.info("Trades table columns:")
        for col in trades_columns:
            logger.info(f"  {col[0]}: {col[1]}")
        
        # Test the executed_trades table structure
        logger.info("\nChecking executed_trades table structure...")
        cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'executed_trades'")
        executed_columns = cursor.fetchall()
        logger.info("Executed_trades table columns:")
        for col in executed_columns:
            logger.info(f"  {col[0]}: {col[1]}")
        
        # Check if the trading_pair matches database formatting
        logger.info(f"\nTrading pair from config: {trading_pair}")
        cursor.execute("SELECT DISTINCT trading_pair FROM trades")
        trades_pairs = cursor.fetchall()
        logger.info(f"Distinct trading pairs in trades table: {trades_pairs}")
        
        cursor.execute("SELECT DISTINCT trading_pair FROM executed_trades")
        executed_pairs = cursor.fetchall()
        logger.info(f"Distinct trading pairs in executed_trades table: {executed_pairs}")
        
        # Try the query but simplify it to just show what's in executed_trades
        logger.info("\nTesting simplified query from executed_trades:")
        simple_query = """
            SELECT 
                'executed_trades' as source,
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
                td.decision
            FROM executed_trades et
            LEFT JOIN trade_decisions td ON et.decision_id = td.id
            WHERE et.trading_pair = %s
            ORDER BY timestamp DESC
            LIMIT 10
        """
        
        df = pd.read_sql_query(simple_query, conn, params=[trading_pair])
        logger.info(f"Found {len(df)} executed trades")
        logger.info("First few rows:")
        logger.info(df.head())
        
        # Now try the combined query with debug info
        logger.info("\nTesting full combined query:")
        full_query = """
            WITH combined_trades AS (
                -- Get trades from the trades table
                SELECT 
                    'trades' as source,
                    timestamp, 
                    trading_pair, 
                    direction, 
                    price, 
                    size, 
                    action
                FROM trades
                WHERE trading_pair = %s
                
                UNION ALL
                
                -- Get trades from the executed_trades table
                SELECT 
                    'executed_trades' as source,
                    et.transaction_time as timestamp, 
                    et.trading_pair, 
                    et.side as direction, 
                    et.average_fill_price as price, 
                    et.filled_quantity as size, 
                    CASE 
                        WHEN td.decision LIKE 'ENTER%%' THEN 'OPEN'
                        WHEN td.decision LIKE 'EXIT%%' THEN 'CLOSE'
                        ELSE 'UNKNOWN'
                    END as action
                FROM executed_trades et
                LEFT JOIN trade_decisions td ON et.decision_id = td.id
                WHERE et.trading_pair = %s
            )
            SELECT * FROM combined_trades
            ORDER BY timestamp DESC
            LIMIT 10
        """
        
        df = pd.read_sql_query(full_query, conn, params=[trading_pair, trading_pair])
        logger.info(f"Found {len(df)} combined trades")
        logger.info(f"Trades from 'trades' table: {len(df[df['source'] == 'trades'])}")
        logger.info(f"Trades from 'executed_trades' table: {len(df[df['source'] == 'executed_trades'])}")
        logger.info("Results:")
        pd.set_option('display.max_columns', None)
        logger.info(df)
        
    except Exception as e:
        logger.error(f"Error executing queries: {str(e)}")
        return 1
    finally:
        if conn:
            db_logger._return_connection(conn)
        await db_logger.close()
    
    return 0

if __name__ == "__main__":
    asyncio.run(test_combined_query()) 