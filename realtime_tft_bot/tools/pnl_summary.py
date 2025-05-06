#!/usr/bin/env python3
"""
TFT Bot PnL Summary Tool

A simple command-line tool to display PnL statistics for the TFT Bot using
the standardized PnL tracking utility.

Usage:
    python tools/pnl_summary.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--verbose]
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import psycopg2

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pnl_tracker import calculate_pnl_stats, print_pnl_summary
from config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Display PnL statistics for TFT Bot')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD) for filtering trades')
    parser.add_argument('--end', help='End date (YYYY-MM-DD) for filtering trades')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed output')
    return parser.parse_args()

def format_date(date_str):
    """Convert YYYY-MM-DD to ISO format with time"""
    if not date_str:
        return None
    try:
        if 'T' in date_str:  # Already has time component
            return date_str
        # Add time component (start of day for start date, end of day for end date)
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.isoformat()
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.")
        sys.exit(1)

def main():
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Format dates
    start_time = format_date(args.start)
    end_time = format_date(args.end)
    
    # Load config
    config = load_config()
    
    # Connect to database
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=config['DB_HOST'],
            port=config['DB_PORT'],
            database=config['DB_NAME'],
            user=config['DB_USER'],
            password=config['DB_PASSWORD']
        )
        
        # Debug: Check raw data
        if args.verbose:
            # Check how many rows we're getting back
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM trade_decisions")
                td_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM executed_trades")
                et_count = cursor.fetchone()[0]
                
                print(f"\nDEBUG: Found {td_count} trade decisions and {et_count} executed trades in the database")
                
                cursor.execute("SELECT id, decision, decision_timestamp, target_price FROM trade_decisions LIMIT 5")
                decisions = cursor.fetchall()
                
                print("\nSample Trade Decisions:")
                for d in decisions:
                    print(f"ID: {d[0]}, Decision: {d[1]}, Time: {d[2]}, Target Price: {d[3]}")
                
                cursor.execute("SELECT id, decision_id, side, average_fill_price, filled_quantity, status FROM executed_trades LIMIT 5")
                trades = cursor.fetchall()
                
                print("\nSample Executed Trades:")
                for t in trades:
                    print(f"ID: {t[0]}, Decision ID: {t[1]}, Side: {t[2]}, Price: {t[3]}, Quantity: {t[4]}, Status: {t[5]}")
        
        # Calculate stats
        stats = calculate_pnl_stats(conn, start_time, end_time)
        
        # Debug: Print raw trades data
        if args.verbose:
            print("\nRaw Trades Data:")
            print(f"Shape: {stats['trades_df'].shape}")
            print(stats['trades_df'].head())
            
            print("\nDirection Counts:")
            print(stats['trades_df']['direction'].value_counts())
            
            # Extract ENTER_LONG and EXIT_LONG trades
            enter_trades = stats['trades_df'][stats['trades_df']['direction'] == 'ENTER_LONG']
            exit_trades = stats['trades_df'][stats['trades_df']['direction'] == 'EXIT_LONG']
            
            print(f"\nFound {len(enter_trades)} ENTER_LONG and {len(exit_trades)} EXIT_LONG trades")
            
            print("\nFirst 3 ENTER_LONG trades:")
            print(enter_trades[['timestamp', 'symbol', 'target_price']].head(3))
            
            print("\nFirst 3 EXIT_LONG trades:")
            print(exit_trades[['timestamp', 'symbol', 'target_price']].head(3))
            
            print("\nPaired Trades:")
            if 'paired_df' in stats and not stats['paired_df'].empty:
                print(f"Shape: {stats['paired_df'].shape}")
                print(stats['paired_df'].head())
            else:
                print("No paired trades found!")
        
        # Print summary
        print_pnl_summary(stats)
        
        # Print detailed results if verbose
        if args.verbose and not stats['detailed_results'].empty:
            print("\nDetailed Trade Results:")
            pd_options = {
                'display.max_rows': None,
                'display.max_columns': None,
                'display.width': None,
                'display.float_format': '{:.8f}'.format
            }
            with pd.option_context(*[item for sublist in pd_options.items() for item in sublist]):
                print(stats['detailed_results'].sort_values('exit_time'))
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 