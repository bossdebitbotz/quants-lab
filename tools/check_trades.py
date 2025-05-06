#!/usr/bin/env python3
import sys
import os
import asyncio

# Add parent to path
sys.path.insert(0, os.path.abspath('..'))

from utils.db_logger import DBLogger
from config import load_config

async def check_trades():
    config = load_config()
    db_logger = DBLogger(config)
    
    if not await db_logger.initialize():
        print('Failed to initialize DB')
        return
    
    try:
        conn = db_logger._get_connection()
        if not conn:
            print('Failed to get connection')
            return
        
        try:
            cursor = conn.cursor()
            
            # Check trades
            cursor.execute('SELECT COUNT(*) FROM trades')
            count = cursor.fetchone()[0]
            print(f'Total trades: {count}')
            
            if count > 0:
                cursor.execute('SELECT id, timestamp, trading_pair, action, direction, price, size, pnl FROM trades ORDER BY timestamp DESC LIMIT 5')
                print("\nRecent trades:")
                for row in cursor.fetchall():
                    print(f'ID: {row[0]}, Time: {row[1]}, Pair: {row[2]}, Action: {row[3]}, Direction: {row[4]}, Price: {row[5]}, Size: {row[6]}, PnL: {row[7]}')
            else:
                print('No trades found')
                
            # Check position state
            cursor.execute('SELECT * FROM position_state')
            print("\nPositions:")
            for row in cursor.fetchall():
                print(f'Position record: {row}')
                
        finally:
            db_logger._return_connection(conn)
    finally:
        await db_logger.close()

if __name__ == "__main__":
    asyncio.run(check_trades()) 