#!/usr/bin/env python3
"""
TFT Bot PnL Summary Tool

A simple command-line tool to display PnL statistics for the TFT Bot using
the standardized PnL tracking utility.

Usage:
    python tools/pnl_summary.py --db path/to/database.db [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import argparse
import sqlite3
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pnl_tracker import calculate_pnl_stats, print_pnl_summary

def parse_args():
    parser = argparse.ArgumentParser(description='Display PnL statistics for TFT Bot')
    parser.add_argument('--db', required=True, help='Path to the SQLite database')
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
    
    # Connect to database
    try:
        conn = sqlite3.connect(args.db)
        
        # Calculate stats
        stats = calculate_pnl_stats(conn, start_time, end_time)
        
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
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 