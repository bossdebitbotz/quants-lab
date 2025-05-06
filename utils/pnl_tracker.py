"""
Standardized PnL Tracking Utility

This module provides standardized functions for calculating PnL (Profit and Loss)
and trade statistics for the TFT Bot. It ensures consistent handling of:
- Scientific notation in price data
- Trade pairing logic
- Filtering criteria for valid trades
- Win rate calculation

Usage:
    from utils.pnl_tracker import calculate_pnl_stats
    stats = calculate_pnl_stats(db_connection)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def fetch_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Fetch all trades from the database with proper numeric conversion.
    
    Args:
        conn: SQLite database connection
        
    Returns:
        DataFrame containing properly formatted trade data
    """
    query = """
    SELECT 
        td.id as decision_id,
        td.timestamp,
        td.symbol,
        td.direction,
        td.target_price,
        td.position_size,
        et.trade_id,
        et.fill_price,
        et.fill_quantity,
        et.timestamp as execution_time,
        et.status
    FROM 
        trade_decisions td
    LEFT JOIN 
        executed_trades et ON td.id = et.decision_id
    ORDER BY 
        td.timestamp
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Convert prices from scientific notation to float
    for col in ['target_price', 'fill_price']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: float(x) if x and x.lower() != 'none' and x.lower() != 'null' else None
            )
    
    return df

def pair_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair entry and exit trades for the same symbol.
    
    Args:
        trades_df: DataFrame of trades from fetch_trades()
        
    Returns:
        DataFrame with paired trades (entry and exit rows)
    """
    paired_trades = []
    
    # Group by symbol
    for symbol, group in trades_df.groupby('symbol'):
        # Sort by timestamp
        sorted_trades = group.sort_values('timestamp')
        
        # Track current position
        position = 0
        entry_trade = None
        
        for _, trade in sorted_trades.iterrows():
            direction = trade['direction']
            
            # Handle entry
            if (position == 0 and direction == 'BUY') or (position == 0 and direction == 'SELL'):
                position = 1 if direction == 'BUY' else -1
                entry_trade = trade
            
            # Handle exit - opposite of current position
            elif (position == 1 and direction == 'SELL') or (position == -1 and direction == 'BUY'):
                if entry_trade is not None:
                    # Create a pair record
                    paired_trades.append({
                        'symbol': symbol,
                        'entry_time': entry_trade['timestamp'],
                        'exit_time': trade['timestamp'],
                        'entry_direction': entry_trade['direction'],
                        'entry_price': _get_best_price(entry_trade),
                        'exit_price': _get_best_price(trade),
                        'position_size': entry_trade['position_size'],
                        'entry_id': entry_trade['decision_id'],
                        'exit_id': trade['decision_id'],
                        'entry_fill_status': entry_trade['status'],
                        'exit_fill_status': trade['status']
                    })
                    
                    # Reset for next trade
                    position = 0
                    entry_trade = None
    
    if paired_trades:
        return pd.DataFrame(paired_trades)
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'symbol', 'entry_time', 'exit_time', 'entry_direction', 
            'entry_price', 'exit_price', 'position_size', 
            'entry_id', 'exit_id', 'entry_fill_status', 'exit_fill_status'
        ])

def _get_best_price(trade: pd.Series) -> float:
    """
    Get the best available price for a trade (fill_price or target_price).
    
    Args:
        trade: A Series representing a single trade
        
    Returns:
        The best available price as a float
    """
    # Use fill_price if available, otherwise use target_price
    if pd.notna(trade.get('fill_price')):
        return float(trade['fill_price'])
    elif pd.notna(trade.get('target_price')):
        return float(trade['target_price'])
    else:
        logger.warning(f"No price found for trade {trade.get('decision_id', 'unknown')}")
        return 0.0

def calculate_pair_pnl(pair: pd.Series) -> Tuple[float, bool]:
    """
    Calculate PnL for a trade pair.
    
    Args:
        pair: A Series representing a paired trade (entry and exit)
        
    Returns:
        Tuple of (pnl_amount, is_winner)
    """
    if not pair['entry_price'] or not pair['exit_price']:
        return 0.0, False
    
    entry_price = float(pair['entry_price'])
    exit_price = float(pair['exit_price'])
    position_size = float(pair['position_size']) if pd.notna(pair['position_size']) else 1.0
    
    if pair['entry_direction'] == 'BUY':
        pnl = (exit_price - entry_price) * position_size
        is_winner = exit_price > entry_price
    else:  # SELL
        pnl = (entry_price - exit_price) * position_size
        is_winner = entry_price > exit_price
    
    return pnl, is_winner

def calculate_pnl_stats(conn: sqlite3.Connection, 
                        start_time: Optional[str] = None, 
                        end_time: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive PnL statistics.
    
    Args:
        conn: SQLite database connection
        start_time: Optional start time for filtering (ISO format)
        end_time: Optional end time for filtering (ISO format)
        
    Returns:
        Dictionary with PnL statistics
    """
    # Get all trades
    trades_df = fetch_trades(conn)
    
    # Filter by time if provided
    if start_time:
        trades_df = trades_df[trades_df['timestamp'] >= start_time]
    if end_time:
        trades_df = trades_df[trades_df['timestamp'] <= end_time]
    
    # Pair trades
    paired_df = pair_trades(trades_df)
    
    # Calculate PnL for each pair
    results = []
    total_pnl = 0.0
    winning_trades = 0
    total_trades = len(paired_df)
    
    for _, pair in paired_df.iterrows():
        pnl, is_winner = calculate_pair_pnl(pair)
        total_pnl += pnl
        
        if is_winner:
            winning_trades += 1
        
        results.append({
            'symbol': pair['symbol'],
            'entry_time': pair['entry_time'],
            'exit_time': pair['exit_time'],
            'entry_direction': pair['entry_direction'],
            'entry_price': pair['entry_price'],
            'exit_price': pair['exit_price'],
            'position_size': pair['position_size'],
            'pnl': pnl,
            'is_winner': is_winner
        })
    
    # Create detailed results DataFrame
    detailed_results = pd.DataFrame(results) if results else pd.DataFrame()
    
    # Calculate win rate
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    # Return comprehensive statistics
    return {
        'total_pnl': round(total_pnl, 8),
        'win_count': winning_trades,
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'detailed_results': detailed_results,
        'by_symbol': detailed_results.groupby('symbol').agg({
            'pnl': 'sum',
            'is_winner': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'count'}) if not detailed_results.empty else pd.DataFrame()
    }

def print_pnl_summary(stats: Dict[str, Any]) -> None:
    """
    Print a summary of PnL statistics.
    
    Args:
        stats: Dictionary from calculate_pnl_stats()
    """
    print(f"\n===== TFT Bot PnL Summary =====")
    print(f"Total PnL: {stats['total_pnl']}")
    print(f"Win Rate: {stats['win_rate']}% ({stats['win_count']}/{stats['total_trades']})")
    
    if not stats['by_symbol'].empty:
        print("\nPnL by Symbol:")
        for symbol, row in stats['by_symbol'].iterrows():
            win_rate = (row['is_winner'] / row['count'] * 100) if row['count'] > 0 else 0
            print(f"  {symbol}: {row['pnl']:.8f} | Win Rate: {win_rate:.2f}% ({int(row['is_winner'])}/{int(row['count'])})")

def get_pnl_report(conn: sqlite3.Connection, 
                   start_time: Optional[str] = None, 
                   end_time: Optional[str] = None) -> str:
    """
    Generate a formatted report of PnL statistics.
    
    Args:
        conn: SQLite database connection
        start_time: Optional start time for filtering (ISO format)
        end_time: Optional end time for filtering (ISO format)
        
    Returns:
        Formatted string report
    """
    stats = calculate_pnl_stats(conn, start_time, end_time)
    
    report = [
        "===== TFT Bot PnL Report =====",
        f"Total PnL: {stats['total_pnl']}",
        f"Win Rate: {stats['win_rate']}% ({stats['win_count']}/{stats['total_trades']})",
        ""
    ]
    
    if not stats['by_symbol'].empty:
        report.append("PnL by Symbol:")
        for symbol, row in stats['by_symbol'].iterrows():
            win_rate = (row['is_winner'] / row['count'] * 100) if row['count'] > 0 else 0
            report.append(f"  {symbol}: {row['pnl']:.8f} | Win Rate: {win_rate:.2f}% ({int(row['is_winner'])}/{int(row['count'])})")
    
    return "\n".join(report) 