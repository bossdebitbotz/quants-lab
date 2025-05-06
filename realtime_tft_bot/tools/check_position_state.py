#!/usr/bin/env python3
"""
Position State Diagnostic Tool

This script can be used to check the current position state in the database
and help diagnose issues with position tracking.
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
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("PositionStateDiagnostic")

async def check_position_state(trading_pair=None, set_position=None):
    """Check the current position state in the database"""
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        # If trading pair not specified, use from config
        if not trading_pair:
            trading_pair = config['TRADING_PAIR']
        
        # If setting a new position
        if set_position:
            position, price, size = set_position.split(',')
            logger.info(f"Setting position state to {position} with price {price} and size {size}")
            
            # Update position state
            result = await db_logger.update_position_state(
                trading_pair,
                position,
                entry_price=float(price) if price.lower() != 'none' else None,
                position_size=float(size) if size.lower() != 'none' else None,
                entry_timestamp=datetime.now() if position != 'NONE' else None
            )
            
            if result:
                logger.info(f"Successfully updated position state to {position}")
            else:
                logger.error("Failed to update position state")
        
        # Get current position state
        position_state = await db_logger.get_position_state(trading_pair)
        
        if position_state:
            logger.info(f"Current position state for {trading_pair}:")
            logger.info(f"  Position: {position_state['position']}")
            logger.info(f"  Entry Price: {position_state['entry_price']}")
            logger.info(f"  Position Size: {position_state['position_size']}")
            logger.info(f"  Entry Time: {position_state['entry_timestamp']}")
            logger.info(f"  Holding Periods: {position_state['holding_periods']}")
            logger.info(f"  Last Update: {position_state.get('last_update')}")
        else:
            logger.error(f"No position state found for {trading_pair}")
    
    finally:
        # Close database connection
        await db_logger.close()

def main():
    parser = argparse.ArgumentParser(description="Position State Diagnostic Tool")
    parser.add_argument("--trading-pair", help="Trading pair to check (e.g., 'BTC-USDT')")
    parser.add_argument("--set-position", help="Set position state (format: POSITION,PRICE,SIZE e.g., 'LONG,10000,0.1' or 'NONE,none,none')")
    
    args = parser.parse_args()
    
    asyncio.run(check_position_state(args.trading_pair, args.set_position))

if __name__ == "__main__":
    main() 