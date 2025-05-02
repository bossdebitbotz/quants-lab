#!/usr/bin/env python3
"""
WLD-USDT Order Book Analysis

This script fetches and analyzes the order book for WLD-USDT on Binance Perpetual Futures
directly from the Binance API without additional dependencies.
"""

import aiohttp
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import signal
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# Binance Futures API base URL
BASE_URL = "https://fapi.binance.com"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Global flag for controlling continuous execution
running = True

def handle_exit_signal(sig, frame):
    """Handle exit signals gracefully"""
    global running
    print("\nShutting down gracefully... (This may take a few seconds)")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def setup_database():
    """Set up the database tables if they don't exist."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Create order_book table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book (
                    id SERIAL PRIMARY KEY,
                    trading_pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    last_update_id BIGINT NOT NULL,
                    price DECIMAL(18, 8) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL,
                    order_type VARCHAR(4) NOT NULL,  -- 'BID' or 'ASK'
                    value DECIMAL(18, 8) NOT NULL
                );
            ''')
            
            # Create index on trading_pair and timestamp for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS order_book_pair_time_idx 
                ON order_book (trading_pair, timestamp);
            ''')
            
            # Create trades table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id BIGINT PRIMARY KEY,
                    trading_pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price DECIMAL(18, 8) NOT NULL,
                    quantity DECIMAL(18, 8) NOT NULL, 
                    quote_quantity DECIMAL(18, 8) NOT NULL,
                    is_buyer_maker BOOLEAN NOT NULL,
                    side VARCHAR(4) NOT NULL  -- 'BUY' or 'SELL'
                );
            ''')
            
            # Create index on trades table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS trades_pair_time_idx 
                ON trades (trading_pair, timestamp);
            ''')
            
            # Drop the existing order_book_summary table if it exists but has missing columns
            try:
                cursor.execute('''
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'order_book_summary' AND column_name = 'best_bid';
                ''')
                if not cursor.fetchone():
                    print("Dropping incomplete order_book_summary table to recreate it properly")
                    cursor.execute("DROP TABLE IF EXISTS order_book_summary;")
            except Exception as e:
                print(f"Warning: Error checking order_book_summary structure: {e}")
            
            # Create order_book_summary table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book_summary (
                    id SERIAL PRIMARY KEY,
                    trading_pair VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    best_bid DECIMAL(18, 8) NOT NULL,
                    best_ask DECIMAL(18, 8) NOT NULL,
                    mid_price DECIMAL(18, 8) NOT NULL,
                    spread DECIMAL(18, 8) NOT NULL,
                    spread_pct DECIMAL(18, 8) NOT NULL,
                    bid_quantity DECIMAL(18, 8) NOT NULL,
                    ask_quantity DECIMAL(18, 8) NOT NULL,
                    imbalance DECIMAL(18, 8) NOT NULL
                );
            ''')
            
            # Create index on summary table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS order_book_summary_pair_time_idx 
                ON order_book_summary (trading_pair, timestamp);
            ''')
            
            # Check if last_update_id column exists in order_book table, add it if it doesn't
            try:
                cursor.execute('''
                    ALTER TABLE order_book 
                    ADD COLUMN IF NOT EXISTS last_update_id BIGINT NOT NULL DEFAULT 0;
                ''')
            except Exception as e:
                print(f"Warning: Could not add last_update_id column: {e}")
            
            # Check if is_buyer_maker column exists in trades table, add it if it doesn't
            try:
                cursor.execute('''
                    ALTER TABLE trades 
                    ADD COLUMN IF NOT EXISTS is_buyer_maker BOOLEAN NOT NULL DEFAULT FALSE;
                ''')
            except Exception as e:
                print(f"Warning: Could not add is_buyer_maker column: {e}")
            
        conn.commit()
        print("Database setup completed successfully")
        return True
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_order_book_to_db(order_book, trading_pair):
    """Save order book data to the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Insert bids and asks into order_book table
            for order_type, orders in [('BID', order_book['bids']), ('ASK', order_book['asks'])]:
                if orders is None or len(orders) == 0:
                    continue
                
                # Build bulk insert values
                values = []
                timestamp = datetime.now()
                last_update_id = order_book.get('last_update_id', 0)
                
                for _, row in orders.iterrows():
                    price = float(row['price'])
                    qty = float(row['quantity'])
                    value = price * qty
                    values.append((
                        trading_pair, timestamp, last_update_id, price, qty, order_type, value
                    ))
                
                # Bulk insert into order_book table
                values_str = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s,%s,%s)", v).decode('utf-8') for v in values)
                if values_str:
                    cursor.execute(f"""
                        INSERT INTO order_book (trading_pair, timestamp, last_update_id, price, quantity, order_type, value)
                        VALUES {values_str}
                    """)
            
            # Calculate and insert summary data
            best_bid = float(order_book['bids']['price'].iloc[0]) if not order_book['bids'].empty else 0
            best_ask = float(order_book['asks']['price'].iloc[0]) if not order_book['asks'].empty else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            spread_pct = (spread / mid_price) * 100 if mid_price else 0
            
            # Calculate total quantity at best 10 levels
            bid_quantity = order_book['bids']['quantity'].head(10).sum() if not order_book['bids'].empty else 0
            ask_quantity = order_book['asks']['quantity'].head(10).sum() if not order_book['asks'].empty else 0
            imbalance = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity) if (bid_quantity + ask_quantity) > 0 else 0
            
            # Insert summary data
            cursor.execute("""
                INSERT INTO order_book_summary 
                (trading_pair, timestamp, best_bid, best_ask, mid_price, spread, spread_pct, 
                bid_quantity, ask_quantity, imbalance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                trading_pair, timestamp, best_bid, best_ask, mid_price, spread, spread_pct,
                bid_quantity, ask_quantity, imbalance
            ))
            
            # Commit the transaction
            conn.commit()
            print(f"Order book data saved to database: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            return True
    except Exception as e:
        print(f"Error saving order book to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_trades_to_db(trades, trading_pair):
    """Save trades data to the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Process each trade and insert into the database
            for trade in trades:
                # Extract data from trade object
                trade_id = trade['id']
                price = float(trade['price'])
                quantity = float(trade['qty'])
                quote_quantity = float(trade['quoteQty'])
                is_buyer_maker = trade['isBuyerMaker']
                # Determine side based on isBuyerMaker (BUY = !isBuyerMaker, SELL = isBuyerMaker)
                side = 'SELL' if is_buyer_maker else 'BUY'
                
                # Convert timestamp from milliseconds to datetime
                # If it's already a datetime (pandas Timestamp), convert to Python datetime
                if isinstance(trade['time'], (datetime, pd.Timestamp)):
                    timestamp = trade['time'].to_pydatetime() if isinstance(trade['time'], pd.Timestamp) else trade['time']
                else:
                    # It's a millisecond timestamp as int
                    timestamp = datetime.fromtimestamp(trade['time'] / 1000.0)
                
                # Insert trade data
                cursor.execute("""
                    INSERT INTO trades 
                    (id, trading_pair, timestamp, price, quantity, quote_quantity, is_buyer_maker, side)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    trade_id, trading_pair, timestamp, price, quantity, 
                    quote_quantity, is_buyer_maker, side
                ))
            
            # Commit the transaction
            conn.commit()
            print(f"Trade data saved to database: {len(trades)} trades")
            return True
    except Exception as e:
        print(f"Error saving trades to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

async def get_order_book(trading_pair, limit=20):
    """
    Get order book data directly from Binance Futures API
    
    Args:
        trading_pair: Trading pair in format 'BASE-QUOTE' (e.g., 'WLD-USDT')
        limit: Order book depth (5, 10, 20, 50, 100, 500, 1000)
    """
    print(f"Fetching order book for {trading_pair} with limit={limit}...")
    
    # Convert from internal format (WLD-USDT) to exchange format (WLDUSDT)
    symbol = trading_pair.replace('-', '')
    print(f"Converted trading pair from {trading_pair} to {symbol} for API request")
    
    # Build request URL
    url = f"{BASE_URL}/fapi/v1/depth"
    params = {
        "symbol": symbol,
        "limit": limit
    }
    
    try:
        # Make API request
        print(f"Making API request to {url} with params: {params}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response (status {response.status}): {error_text}")
                    return None
                
                data = await response.json()
                
        print(f"Request successful! Received data with lastUpdateId: {data.get('lastUpdateId')}")
        print(f"Number of bids: {len(data.get('bids', []))}")
        print(f"Number of asks: {len(data.get('asks', []))}")
        
        # Process data into DataFrames
        bids_df = pd.DataFrame(data["bids"], columns=["price", "quantity"], dtype=float)
        asks_df = pd.DataFrame(data["asks"], columns=["price", "quantity"], dtype=float)
        
        order_book = {
            "timestamp": datetime.now(),
            "last_update_id": data["lastUpdateId"],
            "bids": bids_df,
            "asks": asks_df
        }
        
        return order_book
    except Exception as e:
        print(f"Error fetching order book: {type(e).__name__} - {str(e)}")
        return None

async def get_recent_trades(trading_pair, limit=500):
    """
    Get recent trades directly from Binance Futures API
    
    Args:
        trading_pair: Trading pair in format 'BASE-QUOTE' (e.g., 'WLD-USDT')
        limit: Number of trades to fetch (max 1000)
    """
    print(f"Fetching recent trades for {trading_pair} with limit={limit}...")
    
    # Convert from internal format (WLD-USDT) to exchange format (WLDUSDT)
    symbol = trading_pair.replace('-', '')
    
    # Build request URL
    url = f"{BASE_URL}/fapi/v1/trades"
    params = {
        "symbol": symbol,
        "limit": limit
    }
    
    try:
        # Make API request
        print(f"Making API request to {url} with params: {params}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response (status {response.status}): {error_text}")
                    return None
                
                data = await response.json()
        
        print(f"Request successful! Received {len(data)} trades")
        
        # Print the first trade to see its structure
        if data:
            print(f"Sample trade data structure: {data[0]}")
        
        # Process into DataFrame
        trades_df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        trades_df['time'] = pd.to_datetime(trades_df['time'], unit='ms')
        
        # Convert numeric columns
        trades_df['price'] = trades_df['price'].astype(float)
        trades_df['qty'] = trades_df['qty'].astype(float)
        trades_df['quoteQty'] = trades_df['quoteQty'].astype(float)
        
        # Add side information based on isBuyerMaker field
        # In Binance Futures API, isBuyerMaker=True means a SELL order was filled (maker is buying)
        # isBuyerMaker=False means a BUY order was filled (taker is buying)
        if 'isBuyerMaker' in trades_df.columns:
            trades_df['side'] = trades_df['isBuyerMaker'].apply(lambda x: 'SELL' if x else 'BUY')
        elif 'isBuyer' in trades_df.columns:
            trades_df['side'] = trades_df['isBuyer'].apply(lambda x: 'BUY' if x else 'SELL')
        else:
            # If neither field exists, we can't determine side
            print("Warning: Could not determine trade side - 'isBuyerMaker' or 'isBuyer' fields missing")
            trades_df['side'] = 'UNKNOWN'
        
        return trades_df
    
    except Exception as e:
        print(f"Error fetching trades: {type(e).__name__} - {str(e)}")
        return None

def analyze_trades(trades_df):
    """Analyze trade data and print summary stats"""
    if trades_df is None or trades_df.empty:
        print("No trade data to analyze")
        return
    
    # Basic statistics
    print(f"\nTrade Analysis Summary:")
    print(f"Total trades: {len(trades_df)}")
    print(f"Time range: {trades_df['time'].min()} to {trades_df['time'].max()}")
    
    # Buy/Sell ratio
    buys = trades_df[trades_df['side'] == 'BUY']
    sells = trades_df[trades_df['side'] == 'SELL']
    buy_volume = buys['qty'].sum()
    sell_volume = sells['qty'].sum()
    
    print(f"Buy trades: {len(buys)} ({len(buys)/len(trades_df)*100:.1f}%)")
    print(f"Sell trades: {len(sells)} ({len(sells)/len(trades_df)*100:.1f}%)")
    print(f"Buy volume: {buy_volume:.4f} ({buy_volume/(buy_volume+sell_volume)*100:.1f}%)")
    print(f"Sell volume: {sell_volume:.4f} ({sell_volume/(buy_volume+sell_volume)*100:.1f}%)")
    
    # Price statistics
    print(f"\nPrice range: {trades_df['price'].min()} to {trades_df['price'].max()}")
    print(f"VWAP (Volume-Weighted Average Price): {(trades_df['price'] * trades_df['qty']).sum() / trades_df['qty'].sum():.8f}")
    
    # Save to CSV
    filename = "wld_usdt_trades.csv"
    trades_df.to_csv(filename, index=False)
    print(f"Trade data saved to {filename}")
    
    return trades_df

def plot_trades(trades_df):
    """Plot trade data for analysis"""
    if trades_df is None or trades_df.empty:
        print("No trade data to plot")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Trade prices over time
    buys = trades_df[trades_df['side'] == 'BUY']
    sells = trades_df[trades_df['side'] == 'SELL']
    
    ax1.scatter(buys['time'], buys['price'], color='green', s=buys['qty']*50, alpha=0.5, label='Buy')
    ax1.scatter(sells['time'], sells['price'], color='red', s=sells['qty']*50, alpha=0.5, label='Sell')
    
    ax1.set_title("WLD-USDT Recent Trades")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Trade volumes
    # Resample to see volume by minute
    trades_df['volume'] = trades_df['qty']
    volume_by_minute = trades_df.set_index('time').resample('1min')['volume'].sum()
    
    # Buy/sell volume by minute
    buys_by_minute = buys.set_index('time').resample('1min')['qty'].sum()
    sells_by_minute = sells.set_index('time').resample('1min')['qty'].sum()
    
    ax2.bar(volume_by_minute.index, buys_by_minute, color='green', alpha=0.5, label='Buy Volume')
    ax2.bar(volume_by_minute.index, sells_by_minute, color='red', alpha=0.5, label='Sell Volume', bottom=buys_by_minute)
    
    ax2.set_title("Trade Volume by Minute")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("wld_usdt_trades.png")
    print(f"Trade analysis plot saved to wld_usdt_trades.png")
    plt.close()

def plot_order_book(order_book, levels=10):
    """Plot the order book"""
    if order_book is None:
        print("No order book data to plot")
        return
    
    # Limit to specified number of levels
    bids = order_book["bids"].head(levels)
    asks = order_book["asks"].head(levels)
    
    # Calculate mid price
    best_bid = float(bids["price"].iloc[0])
    best_ask = float(asks["price"].iloc[0])
    mid_price = (best_bid + best_ask) / 2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bids (buy orders) - green
    bid_prices = bids["price"].values
    bid_quantities = bids["quantity"].values
    ax.barh(bid_prices, bid_quantities, height=bid_prices*0.001, color='green', alpha=0.5, label='Bids')
    
    # Plot asks (sell orders) - red
    ask_prices = asks["price"].values
    ask_quantities = asks["quantity"].values
    ax.barh(ask_prices, ask_quantities, height=ask_prices*0.001, color='red', alpha=0.5, label='Asks')
    
    # Add a line for the mid price
    ax.axhline(y=mid_price, color='blue', linestyle='-', alpha=0.7, label=f'Mid Price: {mid_price:.4f}')
    
    # Calculate spread
    spread = best_ask - best_bid
    spread_pct = (spread / mid_price) * 100
    
    # Set labels and title
    ax.set_title(f"WLD-USDT Order Book - Spread: {spread:.6f} ({spread_pct:.4f}%)")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Display order book imbalance
    bid_quantity_sum = bids["quantity"].sum()
    ask_quantity_sum = asks["quantity"].sum()
    imbalance = (bid_quantity_sum - ask_quantity_sum) / (bid_quantity_sum + ask_quantity_sum)
    imbalance_text = f"Order Book Imbalance: {imbalance:.4f} ({'Bullish' if imbalance > 0 else 'Bearish'})"
    
    plt.figtext(0.5, 0.01, imbalance_text, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("wld_usdt_order_book.png")
    print(f"Order book plot saved to wld_usdt_order_book.png")
    plt.close()

async def monitor_order_book(trading_pair="WLD-USDT", limit=20, interval=5):
    """
    Continuously monitor the order book and save data to the database
    
    Args:
        trading_pair: Trading pair to monitor (e.g., 'WLD-USDT')
        limit: Order book depth (5, 10, 20, 50, 100, 500, 1000)
        interval: Interval between updates in seconds
    """
    global running
    
    print(f"Starting order book monitoring for {trading_pair} at {interval}s intervals")
    print("Press Ctrl+C to stop the script")
    
    # Set up the database
    setup_successful = setup_database()
    if not setup_successful:
        print("Failed to set up database. Exiting.")
        return
    
    count = 0
    last_trade_time = None
    
    try:
        while running:
            print(f"\n[{datetime.now()}] Fetching data (iteration {count+1})...")
            
            # Fetch order book
            order_book = await get_order_book(trading_pair, limit)
            if order_book is not None:
                # Save to database
                save_order_book_to_db(order_book, trading_pair)
                
                # Occasionally generate a plot (every 10 iterations)
                if count % 10 == 0:
                    plot_order_book(order_book)
            
            # Fetch recent trades since last update
            # For the first request, get all recent trades
            trades_limit = 10 if last_trade_time else 100
            trades_df = await get_recent_trades(trading_pair, trades_limit)
            
            if trades_df is not None and not trades_df.empty:
                # Save trades to database
                save_trades_to_db(trades_df.to_dict('records'), trading_pair)
                
                # Update the last trade time
                last_trade_time = trades_df['time'].max()
                
                # Every 50 iterations, analyze and plot trades
                if count % 50 == 0:
                    analyze_trades(trades_df)
                    plot_trades(trades_df)
            
            count += 1
            print(f"Waiting {interval} seconds until next update...")
            # Use a small sleep interval to check for the running flag more frequently
            for _ in range(interval):
                if not running:
                    break
                await asyncio.sleep(1)
                
    except Exception as e:
        print(f"Error in monitor_order_book: {type(e).__name__} - {str(e)}")
    finally:
        print("Order book monitoring stopped.")

async def main():
    """Main entry point"""
    try:
        # Start monitoring with default settings
        await monitor_order_book()
    except Exception as e:
        print(f"Error in main function: {type(e).__name__} - {str(e)}")
    finally:
        print("Program exited.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 