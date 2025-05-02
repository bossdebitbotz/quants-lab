"""
Database integration module for order book data collection

This module provides functions to save order book data to a PostgreSQL database
for later backtesting and analysis.
"""

import pandas as pd
import datetime
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, DateTime, MetaData

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def create_engine_and_tables():
    """
    Creates the SQLAlchemy engine and necessary database tables.
    
    Returns:
        SQLAlchemy engine or None if connection fails
    """
    try:
        # Create connection string
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        
        # Create engine
        engine = create_engine(connection_string)
        print("Database engine created successfully.")
        
        # Create metadata and tables if they don't exist
        metadata = MetaData()
        
        # Define the order book bids table
        bids_table = Table(
            'order_book_bids', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('connector', String),
            Column('trading_pair', String),
            Column('price', Float),
            Column('amount', Float),
            Column('value', Float)
        )
        
        # Define the order book asks table
        asks_table = Table(
            'order_book_asks', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('connector', String),
            Column('trading_pair', String),
            Column('price', Float),
            Column('amount', Float),
            Column('value', Float)
        )
        
        # Define the order book summary table
        summary_table = Table(
            'order_book_summary', metadata,
            Column('id', Integer, primary_key=True),
            Column('timestamp', DateTime),
            Column('connector', String),
            Column('trading_pair', String),
            Column('bid_price', Float),
            Column('ask_price', Float),
            Column('spread', Float),
            Column('mid_price', Float),
            Column('imbalance', Float)
        )
        
        # Create the tables in the database
        metadata.create_all(engine)
        print("Database tables created or already exist.")
        
        return engine
    
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Continuing without database functionality.")
        return None

def save_to_database(bids_df, asks_df, connector_name, trading_pair, engine):
    """
    Save order book data to the database.
    
    Args:
        bids_df: DataFrame of bids
        asks_df: DataFrame of asks
        connector_name: Name of the connector
        trading_pair: Trading pair
        engine: SQLAlchemy engine
    """
    if engine is None:
        print("Database engine not available. Skipping database save.")
        return
    
    timestamp = datetime.datetime.now()
    
    try:
        # Add metadata to DataFrames
        bids_df = bids_df.copy()
        asks_df = asks_df.copy()
        
        bids_df['timestamp'] = timestamp
        bids_df['connector'] = connector_name
        bids_df['trading_pair'] = trading_pair
        
        asks_df['timestamp'] = timestamp
        asks_df['connector'] = connector_name
        asks_df['trading_pair'] = trading_pair
        
        # Save bids and asks to database
        bids_df.to_sql('order_book_bids', engine, if_exists='append', index=False)
        asks_df.to_sql('order_book_asks', engine, if_exists='append', index=False)
        
        # Calculate and save summary data
        best_bid = bids_df['price'].max()
        best_ask = asks_df['price'].min()
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate imbalance
        bid_value = bids_df.head(10)['value'].sum()
        ask_value = asks_df.head(10)['value'].sum()
        total_value = bid_value + ask_value
        imbalance = (bid_value - ask_value) / total_value if total_value > 0 else 0
        
        summary_data = pd.DataFrame([{
            'timestamp': timestamp,
            'connector': connector_name,
            'trading_pair': trading_pair,
            'bid_price': best_bid,
            'ask_price': best_ask,
            'spread': spread,
            'mid_price': mid_price,
            'imbalance': imbalance
        }])
        
        summary_data.to_sql('order_book_summary', engine, if_exists='append', index=False)
        print(f"Order book data saved to database at {timestamp}")
        
        return timestamp
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None

def query_order_book_data(engine, start_time=None, end_time=None, limit=10):
    """
    Query order book data from the database.
    
    Args:
        engine: SQLAlchemy engine
        start_time: Optional start time for the query
        end_time: Optional end time for the query
        limit: Maximum number of records to return
        
    Returns:
        Tuple of (summary_df, bids_df, asks_df) DataFrames
    """
    if engine is None:
        print("Database engine not available.")
        return None, None, None
    
    try:
        # Build query conditions
        conditions = []
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        # Query summary data
        summary_query = f"SELECT * FROM order_book_summary{where_clause} ORDER BY timestamp DESC{limit_clause}"
        summary_df = pd.read_sql(summary_query, engine)
        
        if summary_df.empty:
            print("No summary data found for the specified time range.")
            return None, None, None
        
        # Get the most recent timestamp
        if not start_time and not end_time and not summary_df.empty:
            latest_timestamp = summary_df['timestamp'].iloc[0]
            
            # Query the corresponding bids and asks
            bids_query = f"SELECT * FROM order_book_bids WHERE timestamp = '{latest_timestamp}'"
            asks_query = f"SELECT * FROM order_book_asks WHERE timestamp = '{latest_timestamp}'"
            
            bids_df = pd.read_sql(bids_query, engine)
            asks_df = pd.read_sql(asks_query, engine)
            
            print(f"Retrieved data from timestamp: {latest_timestamp}")
            print(f"Summary records: {len(summary_df)}")
            print(f"Bid records: {len(bids_df)}")
            print(f"Ask records: {len(asks_df)}")
            
            return summary_df, bids_df, asks_df
        else:
            print(f"Retrieved {len(summary_df)} summary records.")
            return summary_df, None, None
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return None, None, None

async def monitor_order_book_for_backtesting(connector_name, trading_pair, get_order_book_func, format_func, cumulative_func, interval=60, duration=86400, engine=None):
    """
    Monitor the order book continuously for backtesting.
    
    Args:
        connector_name: Name of the connector
        trading_pair: Trading pair
        get_order_book_func: Function to get order book snapshot
        format_func: Function to format order book data
        cumulative_func: Function to calculate cumulative volume
        interval: Update interval in seconds (default: 60 seconds)
        duration: Total monitoring duration in seconds (default: 86400 seconds = 1 day)
        engine: SQLAlchemy engine for database storage
    """
    import pandas as pd
    import asyncio
    
    # For Binance Perpetual, we need to format the trading pair without a hyphen
    formatted_trading_pair = trading_pair
    if connector_name == 'binance_perpetual':
        formatted_trading_pair = trading_pair.replace('-', '')
    
    start_time = pd.Timestamp.now()
    print(f"Starting order book monitoring at {start_time}")
    print(f"Monitoring will run for {duration//3600} hours with {interval} second intervals")
    print(f"Expected completion at {start_time + pd.Timedelta(seconds=duration)}")
    
    iterations = duration // interval
    successful_captures = 0
    
    for i in range(iterations):
        try:
            current_time = pd.Timestamp.now()
            elapsed = (current_time - start_time).total_seconds()
            remaining = duration - elapsed
            
            print(f"Capture {i+1}/{iterations} - Elapsed: {elapsed:.0f}s, Remaining: {remaining:.0f}s")
            
            order_book = await get_order_book_func(connector_name, formatted_trading_pair)
            bids_df, asks_df = format_func(order_book)
            bids_df = cumulative_func(bids_df)
            asks_df = cumulative_func(asks_df)
            
            # Save to database
            if engine is not None:
                save_to_database(bids_df, asks_df, connector_name, trading_pair, engine)
                
            successful_captures += 1
            
            # Print a summary every 10 captures
            if i % 10 == 0:
                best_bid = bids_df['price'].max()
                best_ask = asks_df['price'].min()
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                
                print(f"  Mid Price: {mid_price:.6f}, Spread: {spread:.6f}")
                print(f"  Successful captures: {successful_captures}/{i+1}")
            
            # Wait for the next interval
            if i < iterations - 1:
                await asyncio.sleep(interval)
                
        except Exception as e:
            print(f"Error in capture {i+1}: {type(e).__name__} - {e}")
            print("Continuing with next capture...")
            await asyncio.sleep(5)  # Short wait before retry
    
    end_time = pd.Timestamp.now()
    print(f"Order book monitoring completed at {end_time}")
    print(f"Total duration: {(end_time - start_time).total_seconds():.0f} seconds")
    print(f"Successful captures: {successful_captures}/{iterations} ({successful_captures/iterations*100:.1f}%)")
    
    return successful_captures 