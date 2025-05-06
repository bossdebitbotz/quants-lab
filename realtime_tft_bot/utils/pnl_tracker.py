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
import psycopg2
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def fetch_trades(conn: Union[sqlite3.Connection, psycopg2.extensions.connection]) -> pd.DataFrame:
    """
    Fetch all trades from the database with proper numeric conversion.
    
    Args:
        conn: Database connection (SQLite or PostgreSQL)
        
    Returns:
        DataFrame containing properly formatted trade data
    """
    query = """
    SELECT 
        td.id as decision_id,
        td.decision_timestamp as timestamp,
        td.trading_pair as symbol,
        td.decision as direction,
        td.target_price,
        td.prediction_value,
        et.id as trade_id,
        et.average_fill_price as fill_price,
        et.filled_quantity as fill_quantity,
        et.transaction_time as execution_time,
        et.status,
        et.side as trade_side
    FROM 
        trade_decisions td
    LEFT JOIN 
        executed_trades et ON td.id = et.decision_id
    ORDER BY 
        td.decision_timestamp
    """
    
    try:
        # Check connection type to handle parameter placeholders appropriately
        if isinstance(conn, sqlite3.Connection):
            # SQLite uses ? placeholders
            df = pd.read_sql_query(query, conn)
        else:
            # PostgreSQL uses %s placeholders
            df = pd.read_sql_query(query, conn)
        
        # Convert prices from scientific notation to float
        for col in ['target_price', 'fill_price', 'prediction_value', 'fill_quantity']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: float(x) if x and x.lower() != 'none' and x.lower() != 'null' else None
                )
        
        # Add a default position_size for compatibility with the rest of the code
        df['position_size'] = 0.01  # Default from config
        
        # Handle scientific notation values (0E-8)
        df['fill_price'] = df['fill_price'].apply(lambda x: 0.0 if x == 0 else x)
        df['fill_quantity'] = df['fill_quantity'].apply(lambda x: 0.0 if x == 0 else x)
        
        # For missing fill prices, use target price
        mask = (df['fill_price'].isna()) | (df['fill_price'] == 0)
        df.loc[mask, 'fill_price'] = df.loc[mask, 'target_price']
        
        # Default quantity for missing values
        df.loc[df['fill_quantity'].isna() | (df['fill_quantity'] == 0), 'fill_quantity'] = 116.0
        
        return df
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'decision_id', 'timestamp', 'symbol', 'direction', 
            'target_price', 'prediction_value', 'position_size', 'trade_id', 
            'fill_price', 'fill_quantity', 'execution_time', 'status', 'trade_side'
        ])

def pair_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair ENTER and EXIT trades based on direction and chronological order.
    
    Args:
        trades_df: DataFrame of trades with columns decision_id, timestamp, direction, etc.
        
    Returns:
        DataFrame with paired trades
    """
    if trades_df.empty:
        logger.warning("No trades to pair")
        return pd.DataFrame()
    
    # Sort by timestamp
    sorted_df = trades_df.sort_values('timestamp')
    
    # Group by trading pair
    pairs = []
    
    # Group by symbol
    for symbol, symbol_trades in sorted_df.groupby('symbol'):
        # Process chronologically
        open_position = None
        
        for _, trade in symbol_trades.iterrows():
            direction = trade['direction']
            
            # Skip trades that don't have direction
            if pd.isna(direction):
                continue
                
            # Handle entry trades
            if 'ENTER' in direction and open_position is None:
                # Store the entry trade
                open_position = {
                    'symbol': symbol,
                    'entry_decision_id': trade['decision_id'],
                    'entry_trade_id': trade['trade_id'],
                    'entry_direction': direction,
                    'entry_price': trade['target_price'],  # Use target price as default
                    'position_size': trade['position_size'],
                    'entry_time': trade['timestamp']
                }
                
                # If we have fill_price from executed trades, use that
                if not pd.isna(trade['fill_price']) and trade['fill_price'] > 0:
                    open_position['entry_price'] = trade['fill_price']
            
            # Handle exit trades when we have an open position
            elif 'EXIT' in direction and open_position is not None:
                # Check for compatible direction (EXIT_LONG after ENTER_LONG)
                entry_long = 'LONG' in open_position['entry_direction']
                exit_long = 'LONG' in direction
                entry_short = 'SHORT' in open_position['entry_direction']
                exit_short = 'SHORT' in direction
                
                if (entry_long and exit_long) or (entry_short and exit_short):
                    # Create a complete position pair
                    position = open_position.copy()
                    position.update({
                        'exit_decision_id': trade['decision_id'],
                        'exit_trade_id': trade['trade_id'],
                        'exit_direction': direction,
                        'exit_price': trade['target_price'],  # Use target price as default
                        'exit_time': trade['timestamp']
                    })
                    
                    # If we have fill_price from executed trades, use that
                    if not pd.isna(trade['fill_price']) and trade['fill_price'] > 0:
                        position['exit_price'] = trade['fill_price']
                    
                    # Add to pairs
                    pairs.append(position)
                    open_position = None
                else:
                    logger.warning(f"Direction mismatch: {open_position['entry_direction']} vs {direction}")
    
    # Create DataFrame from pairs
    if not pairs:
        logger.warning("No valid trade pairs found")
        return pd.DataFrame()
    
    paired_df = pd.DataFrame(pairs)
    logger.info(f"Paired {len(paired_df)} trades")
    
    return paired_df

def calculate_pair_pnl(pair: pd.Series) -> Tuple[float, bool]:
    """
    Calculate PnL for a pair of trades.
    
    Args:
        pair: Series from paired_df with entry and exit data
        
    Returns:
        Tuple of (pnl_value, is_winner)
    """
    entry_price = pair['entry_price']
    exit_price = pair['exit_price']
    position_size = pair['position_size']
    
    if pd.isna(entry_price) or pd.isna(exit_price) or pd.isna(position_size):
        logger.warning(f"Missing data for PnL calculation: entry={entry_price}, exit={exit_price}, size={position_size}")
        return 0.0, False
    
    # Determine if long or short based on entry direction
    is_long = 'LONG' in pair['entry_direction']
    
    # Calculate PnL
    if is_long:
        # Long position: (exit - entry) * size
        pnl = (exit_price - entry_price) * position_size
    else:
        # Short position: (entry - exit) * size
        pnl = (entry_price - exit_price) * position_size
    
    # Check if it's a winner
    is_winner = pnl > 0
    
    return pnl, is_winner

def calculate_pnl_stats(conn: Union[sqlite3.Connection, psycopg2.extensions.connection], 
                        start_time: Optional[str] = None, 
                        end_time: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive PnL statistics.
    
    Args:
        conn: Database connection (SQLite or PostgreSQL)
        start_time: Optional start time for filtering (ISO format)
        end_time: Optional end time for filtering (ISO format)
        
    Returns:
        Dictionary with PnL statistics
    """
    # Get all trades
    trades_df = fetch_trades(conn)
    
    # Log trade count
    logger.info(f"Fetched {len(trades_df)} trades from database")
    
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
        }).rename(columns={'symbol': 'count'}) if not detailed_results.empty else pd.DataFrame(),
        # Include raw data for debugging
        'trades_df': trades_df,
        'paired_df': paired_df
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

def get_pnl_report(conn: Union[sqlite3.Connection, psycopg2.extensions.connection], 
                   start_time: Optional[str] = None, 
                   end_time: Optional[str] = None) -> str:
    """
    Generate a formatted report of PnL statistics.
    
    Args:
        conn: Database connection (SQLite or PostgreSQL)
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