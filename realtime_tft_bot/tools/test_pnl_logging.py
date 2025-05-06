#!/usr/bin/env python3
"""
PnL Logging Test Script

This script simulates trade logging to verify that PnL tracking works properly.
It creates sample OPEN and CLOSE trades and logs them to the database.
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
import random

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

logger = logging.getLogger("PnLTest")

async def create_test_trades(db_logger, trading_pair, num_positions=5, win_rate=0.6):
    """
    Create test trades with realistic PnL values
    
    Args:
        db_logger: Database logger instance
        trading_pair: Trading pair to use
        num_positions: Number of test positions to create
        win_rate: Percentage of winning trades
        
    Returns:
        List of trade IDs
    """
    base_price = 0.91  # Starting price
    trade_ids = []
    
    # Create trades over the past few days
    for i in range(num_positions):
        # Random direction (60% long, 40% short)
        direction = "BUY" if random.random() < 0.6 else "SELL"
        
        # Randomize entry price around the base
        price_change = random.uniform(-0.02, 0.02)
        entry_price = base_price * (1 + price_change)
        
        # Random position size
        size = round(random.uniform(0.01, 0.05), 4)
        
        # Calculate entry fees
        entry_fees = entry_price * size * 0.0001
        
        # Calculate timestamp (starting from 5 days ago)
        entry_time = datetime.now() - timedelta(days=5, minutes=i*180)
        
        # Log OPEN trade
        logger.info(f"Creating OPEN {direction} trade for {trading_pair} at price {entry_price}")
        entry_id = await db_logger.log_trade(
            timestamp=entry_time,
            trading_pair=trading_pair,
            direction=direction,
            price=entry_price,
            size=size,
            action="OPEN",
            order_type="MARKET",
            status="FILLED",
            pnl=0.0,
            fees=entry_fees
        )
        
        trade_ids.append(entry_id)
        
        # Random holding period (1-24 hours)
        holding_hours = random.randint(1, 24)
        exit_time = entry_time + timedelta(hours=holding_hours)
        
        # Determine if this is a winning trade based on win rate
        is_winning = random.random() < win_rate
        
        # Calculate exit price based on win/loss
        if direction == "BUY":  # LONG position
            if is_winning:
                # Winning long: price goes up
                price_change = random.uniform(0.005, 0.03)
            else:
                # Losing long: price goes down
                price_change = random.uniform(-0.03, -0.001)
            
            exit_price = entry_price * (1 + price_change)
            pnl = (exit_price - entry_price) * size
            
        else:  # SHORT position
            if is_winning:
                # Winning short: price goes down
                price_change = random.uniform(-0.03, -0.001)
            else:
                # Losing short: price goes up
                price_change = random.uniform(0.001, 0.03)
                
            exit_price = entry_price * (1 + price_change)
            pnl = (entry_price - exit_price) * size
        
        # Calculate exit fees
        exit_fees = exit_price * size * 0.0001
        
        # Log CLOSE trade
        logger.info(f"Creating CLOSE trade for {trading_pair} at price {exit_price} with PnL {pnl}")
        exit_id = await db_logger.log_trade(
            timestamp=exit_time,
            trading_pair=trading_pair,
            direction="SELL" if direction == "BUY" else "BUY",  # Opposite direction to close
            price=exit_price,
            size=size,
            action="CLOSE",
            order_type="MARKET",
            status="FILLED",
            pnl=pnl,
            fees=exit_fees
        )
        
        trade_ids.append(exit_id)
        
        # Update base price for next trade
        base_price = exit_price
    
    return trade_ids

async def main():
    parser = argparse.ArgumentParser(description="PnL Logging Test Tool")
    parser.add_argument("--pair", type=str, help="Trading pair to use (default: from config)")
    parser.add_argument("--positions", type=int, default=5, help="Number of test positions to create")
    parser.add_argument("--win-rate", type=float, default=0.6, help="Percentage of winning trades (0-1)")
    parser.add_argument("--clean", action="store_true", help="Clean existing test data before creating new trades")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Use trading pair from config if not specified
    trading_pair = args.pair or config.get('TRADING_PAIR', 'WLD-USDT')
    
    # Initialize database logger
    db_logger = DBLogger(config)
    await db_logger.initialize()
    
    try:
        # Clean existing test data if requested
        if args.clean:
            logger.info("Cleaning existing test data...")
            conn = db_logger._get_connection()
            if conn:
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("DELETE FROM trades WHERE trading_pair = %s", (trading_pair,))
                        cursor.execute("DELETE FROM performance_metrics WHERE trading_pair = %s", (trading_pair,))
                        conn.commit()
                        logger.info("Test data cleaned successfully")
                except Exception as e:
                    logger.error(f"Error cleaning test data: {str(e)}")
                    conn.rollback()
                finally:
                    db_logger._return_connection(conn)
        
        # Create test trades
        logger.info(f"Creating {args.positions} test positions for {trading_pair} with {args.win_rate*100}% win rate...")
        trade_ids = await create_test_trades(
            db_logger, 
            trading_pair, 
            num_positions=args.positions,
            win_rate=args.win_rate
        )
        
        logger.info(f"Created {len(trade_ids)} trades")
        logger.info("Run pnl_tracker.py to view the results")
    
    finally:
        # Close database connections
        await db_logger.close()

if __name__ == "__main__":
    asyncio.run(main()) 