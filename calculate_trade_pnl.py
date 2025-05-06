#!/usr/bin/env python3
"""
Calculate Trade PnL

This script pairs OPEN and CLOSE trades from both database tables
and calculates PnL for each completed trade cycle.
"""

import os
import sys
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add the realtime_tft_bot directory to the path for imports
sys.path.insert(0, os.path.abspath('realtime_tft_bot'))

from utils.db_logger import DBLogger
from config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CalculatePnL")

async def calculate_trade_pnl():
    """Calculate PnL for all trade pairs"""
    
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection")
        return 1
    
    try:
        # Get all trades from both sources
        conn = db_logger._get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return 1
        
        try:
            # Extract the trading pair components for different formats
            trading_pair = config.get('TRADING_PAIR', 'WLD-USDT')
            pair_components = trading_pair.split('-')
            exchange_format = ''.join(pair_components)
            
            # Query to get all trades from both tables
            query = """
                WITH all_trades AS (
                    -- Get trades from trades table
                    SELECT 
                        id,
                        'trades' as source,
                        timestamp, 
                        trading_pair, 
                        direction, 
                        price, 
                        size, 
                        action,
                        status,
                        order_type,
                        pnl as recorded_pnl,
                        fees
                    FROM trades
                    WHERE trading_pair = %s
                    
                    UNION ALL
                    
                    -- Get trades from executed_trades with trade_decisions
                    SELECT 
                        et.id,
                        'executed_trades' as source,
                        et.transaction_time as timestamp, 
                        et.trading_pair, 
                        et.side as direction, 
                        COALESCE(et.average_fill_price, td.target_price) as price, 
                        COALESCE(et.filled_quantity, et.requested_quantity) as size, 
                        CASE 
                            WHEN td.decision LIKE 'ENTER%%' THEN 'OPEN'
                            WHEN td.decision LIKE 'EXIT%%' THEN 'CLOSE'
                            ELSE 'UNKNOWN'
                        END as action,
                        et.status,
                        et.order_type,
                        NULL as recorded_pnl,
                        et.commission as fees
                    FROM executed_trades et
                    JOIN trade_decisions td ON et.decision_id = td.id
                    WHERE et.trading_pair = %s
                )
                SELECT * FROM all_trades
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(query, conn, params=[trading_pair, exchange_format])
            print(f"Loaded {len(df)} trades from both tables")
            
            if df.empty:
                print("No trades found")
                return 0
            
            # Pair OPEN and CLOSE trades to calculate PnL
            open_trades = []
            trade_pairs = []
            
            for _, trade in df.iterrows():
                if trade['action'] == 'OPEN':
                    # Store open trade
                    open_trades.append(trade)
                elif trade['action'] == 'CLOSE':
                    # Try to match with the oldest open trade with opposite direction
                    matched = False
                    for i, open_trade in enumerate(open_trades):
                        # For a valid pair: BUY-OPEN should close with SELL-CLOSE and vice versa
                        if (open_trade['direction'] == 'BUY' and trade['direction'] == 'SELL') or \
                           (open_trade['direction'] == 'SELL' and trade['direction'] == 'BUY'):
                            # We found a match
                            trade_pair = {
                                'open_id': open_trade['id'],
                                'open_source': open_trade['source'],
                                'open_time': open_trade['timestamp'],
                                'open_direction': open_trade['direction'],
                                'open_price': open_trade['price'],
                                'open_size': open_trade['size'],
                                'open_status': open_trade['status'],
                                'close_id': trade['id'],
                                'close_source': trade['source'],
                                'close_time': trade['timestamp'],
                                'close_direction': trade['direction'],
                                'close_price': trade['price'],
                                'close_size': trade['size'],
                                'close_status': trade['status'],
                                'recorded_pnl': trade['recorded_pnl']
                            }
                            
                            # Calculate holding period
                            if isinstance(open_trade['timestamp'], pd.Timestamp) and isinstance(trade['timestamp'], pd.Timestamp):
                                holding_period = (trade['timestamp'] - open_trade['timestamp']).total_seconds() / 60  # in minutes
                                trade_pair['holding_period_min'] = holding_period
                            
                            # Calculate PnL
                            if open_trade['price'] and trade['price']:
                                if open_trade['direction'] == 'BUY':  # Long position
                                    # For long: PnL = (sell_price - buy_price) * size
                                    trade_pair['calculated_pnl'] = (trade['price'] - open_trade['price']) * min(open_trade['size'] or 0, trade['size'] or 0)
                                else:  # Short position
                                    # For short: PnL = (buy_price - sell_price) * size
                                    trade_pair['calculated_pnl'] = (open_trade['price'] - trade['price']) * min(open_trade['size'] or 0, trade['size'] or 0)
                                    
                                # Also calculate percentage return
                                trade_pair['pnl_percent'] = (trade_pair['calculated_pnl'] / (open_trade['price'] * open_trade['size'])) * 100 if open_trade['price'] and open_trade['size'] else None
                            
                            trade_pairs.append(trade_pair)
                            open_trades.pop(i)  # Remove the matched open trade
                            matched = True
                            break
                    
                    if not matched:
                        print(f"Warning: Found CLOSE trade without matching OPEN: {trade['id']} ({trade['timestamp']})")
            
            if open_trades:
                print(f"Warning: {len(open_trades)} OPEN trades without matching CLOSE")
                for trade in open_trades:
                    print(f"  {trade['id']} ({trade['timestamp']}): {trade['direction']} at {trade['price']}")
            
            # Convert to DataFrame for better analysis
            if trade_pairs:
                pairs_df = pd.DataFrame(trade_pairs)
                
                # Display summary statistics
                total_trades = len(pairs_df)
                winning_trades = len(pairs_df[pairs_df['calculated_pnl'] > 0])
                losing_trades = len(pairs_df[pairs_df['calculated_pnl'] <= 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                print("\n=== Trade PnL Summary ===")
                print(f"Total Completed Trades: {total_trades}")
                print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
                print(f"Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
                
                if not pairs_df.empty:
                    total_pnl = pairs_df['calculated_pnl'].sum()
                    avg_pnl = pairs_df['calculated_pnl'].mean()
                    avg_win = pairs_df.loc[pairs_df['calculated_pnl'] > 0, 'calculated_pnl'].mean() if winning_trades > 0 else 0
                    avg_loss = pairs_df.loc[pairs_df['calculated_pnl'] <= 0, 'calculated_pnl'].mean() if losing_trades > 0 else 0
                    
                    print(f"Total PnL: {total_pnl:.6f}")
                    print(f"Average PnL: {avg_pnl:.6f}")
                    print(f"Average Win: {avg_win:.6f}")
                    print(f"Average Loss: {avg_loss:.6f}")
                    
                    # Calculate drawdown
                    pairs_df['cumulative_pnl'] = pairs_df['calculated_pnl'].cumsum()
                    pairs_df['peak'] = pairs_df['cumulative_pnl'].cummax()
                    pairs_df['drawdown'] = pairs_df['peak'] - pairs_df['cumulative_pnl']
                    max_drawdown = pairs_df['drawdown'].max()
                    print(f"Max Drawdown: {max_drawdown:.6f}")
                
                # Display individual trade details
                print("\n=== Recent Trade Pairs ===")
                print(f"{'Open Time':<20} {'Close Time':<20} {'Direction':<8} {'Open Price':<10} {'Close Price':<10} {'Size':<8} {'PnL':<10} {'% Return':<10}")
                print("-" * 100)
                
                # Show recent trades first
                for _, pair in pairs_df.sort_values('close_time', ascending=False).head(10).iterrows():
                    if isinstance(pair['open_time'], pd.Timestamp):
                        open_time = pair['open_time'].strftime('%Y-%m-%d %H:%M')
                    else:
                        open_time = str(pair['open_time'])
                        
                    if isinstance(pair['close_time'], pd.Timestamp):
                        close_time = pair['close_time'].strftime('%Y-%m-%d %H:%M')
                    else:
                        close_time = str(pair['close_time'])
                    
                    direction = "LONG" if pair['open_direction'] == 'BUY' else "SHORT"
                    open_price = f"{pair['open_price']:.4f}" if pair['open_price'] else "N/A"
                    close_price = f"{pair['close_price']:.4f}" if pair['close_price'] else "N/A"
                    size = f"{pair['open_size']:.4f}" if pair['open_size'] else "N/A"
                    pnl = f"{pair['calculated_pnl']:.6f}" if 'calculated_pnl' in pair else "N/A"
                    pnl_pct = f"{pair['pnl_percent']:.2f}%" if 'pnl_percent' in pair and pair['pnl_percent'] is not None else "N/A"
                    
                    print(f"{open_time:<20} {close_time:<20} {direction:<8} {open_price:<10} {close_price:<10} {size:<8} {pnl:<10} {pnl_pct:<10}")
                
                # Save to CSV
                pairs_df.to_csv('trade_pnl_analysis.csv', index=False)
                print("\nDetailed analysis saved to trade_pnl_analysis.csv")
            else:
                print("No complete trade pairs found")
            
        finally:
            db_logger._return_connection(conn)
    
    finally:
        await db_logger.close()
    
    return 0

if __name__ == "__main__":
    asyncio.run(calculate_trade_pnl()) 