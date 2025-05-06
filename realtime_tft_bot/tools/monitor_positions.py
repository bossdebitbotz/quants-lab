#!/usr/bin/env python3
"""
Position Monitoring Tool

This script continuously monitors the bot's position state and logs changes.
It helps identify synchronization issues by showing the position state changes in real-time.
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db_logger import DBLogger
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("position_monitor.log")
    ]
)

logger = logging.getLogger("PositionMonitor")

async def monitor_positions(polling_interval=5):
    """
    Continuously monitor position state
    
    Args:
        polling_interval: Time between checks in seconds
    """
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        trading_pair = config['TRADING_PAIR']
        logger.info(f"Starting position monitor for {trading_pair}")
        logger.info(f"Polling interval: {polling_interval} seconds")
        
        last_state = None
        
        while True:
            try:
                # Get current position state
                position_state = await db_logger.get_position_state(trading_pair)
                
                if not position_state:
                    logger.warning(f"No position state found for {trading_pair}")
                else:
                    # Check if state has changed
                    if last_state is None or position_state['position'] != last_state['position']:
                        logger.info(f"Position CHANGED: {last_state['position'] if last_state else 'NONE'} -> {position_state['position']}")
                        logger.info(f"New position details: Entry Price: {position_state['entry_price']}, " +
                                   f"Size: {position_state['position_size']}, Time: {position_state['entry_timestamp']}")
                    
                    # Check if holding periods changed
                    elif last_state and position_state['holding_periods'] != last_state['holding_periods']:
                        logger.info(f"Holding periods updated: {last_state['holding_periods']} -> {position_state['holding_periods']}")
                    
                    # Save current state
                    last_state = position_state
                
                # Wait for next check
                await asyncio.sleep(polling_interval)
                
            except Exception as e:
                logger.error(f"Error checking position state: {str(e)}")
                await asyncio.sleep(polling_interval)
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Position Monitoring Tool")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(monitor_positions(args.interval))
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")

if __name__ == "__main__":
    main() 