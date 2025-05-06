#!/usr/bin/env python3
"""
Check Database Tables Script

This script lists all tables in the database and their row counts.
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
logger = logging.getLogger("CheckTables")

async def check_database_tables():
    """List all tables in the database and their row counts"""
    
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
                # Get list of tables
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                
                tables = cursor.fetchall()
                
                if not tables:
                    print("No tables found in the database")
                    return 0
                
                print("\n--- Database Tables ---")
                
                # For each table, get the row count
                for table in tables:
                    table_name = table[0]
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    # Get the most recent row if there are any
                    if count > 0:
                        try:
                            # Look for timestamp-like columns
                            cursor.execute(f"""
                                SELECT column_name 
                                FROM information_schema.columns 
                                WHERE table_name = '{table_name}' 
                                AND (data_type LIKE '%timestamp%' OR column_name LIKE '%time%' OR column_name LIKE '%date%')
                            """)
                            
                            timestamp_columns = cursor.fetchall()
                            
                            if timestamp_columns:
                                time_col = timestamp_columns[0][0]
                                cursor.execute(f"SELECT {time_col} FROM {table_name} ORDER BY {time_col} DESC LIMIT 1")
                                latest = cursor.fetchone()[0]
                                print(f"Table: {table_name} - {count} rows - Latest: {latest}")
                            else:
                                print(f"Table: {table_name} - {count} rows")
                        except Exception as e:
                            # If we can't query the latest timestamp, just show the count
                            print(f"Table: {table_name} - {count} rows")
                    else:
                        print(f"Table: {table_name} - {count} rows")
                
        finally:
            # Return connection to pool
            db_logger._return_connection(conn)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    try:
        asyncio.run(check_database_tables())
    except KeyboardInterrupt:
        print("\nCheck cancelled by user")
    except Exception as e:
        print(f"Error checking tables: {str(e)}")

if __name__ == "__main__":
    main() 