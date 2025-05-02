#!/usr/bin/env python3

import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define cells
cells = [
    nbf.v4.new_markdown_cell("""# WLD-USDT Order Book Analysis

This notebook fetches and analyzes the order book for WLD-USDT on Binance Perpetual."""),
    
    nbf.v4.new_code_cell("""# Install any missing dependencies
import sys
!{sys.executable} -m pip install pandas matplotlib numpy nest_asyncio"""),
    
    nbf.v4.new_code_cell("""# Import libraries
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import nest_asyncio  # This allows asyncio to work in Jupyter

# Enable asyncio in Jupyter
nest_asyncio.apply()

# Import our data source module
from core.data_sources.clob import CLOBDataSource

# Confirm imports worked
print("Libraries imported successfully!")"""),
    
    nbf.v4.new_code_cell("""# Initialize the CLOBDataSource and confirm it works
clob = CLOBDataSource()
print(f"CLOBDataSource initialized with {len(clob.connectors)} connectors")
print(f"Available connectors: {list(clob.connectors.keys())}")

# Verify binance_perpetual connector is available
if 'binance_perpetual' in clob.connectors:
    print("✅ Binance Perpetual connector is available")
else:
    print("❌ Binance Perpetual connector is NOT available")"""),
    
    nbf.v4.new_code_cell("""# Simple function to get order book data
async def get_binance_order_book(trading_pair, limit=20):
    """Get order book data from Binance Perpetual"""
    print(f"Fetching order book for {trading_pair} with limit={limit}...")
    
    # Get connector
    connector = clob.connectors.get("binance_perpetual")
    if not connector:
        print("Error: Binance Perpetual connector not found")
        return None
    
    # Convert from internal format (WLD-USDT) to exchange format (WLDUSDT)
    symbol = trading_pair.replace('-', '')
    print(f"Converted trading pair from {trading_pair} to {symbol} for API request")
    
    try:
        # Make API request to get order book
        print(f"Making API request to /fapi/v1/depth with symbol={symbol}, limit={limit}")
        response = await connector._api_get(
            path_url="/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            is_auth_required=False,
            limit_id="REQUEST_WEIGHT"
        )
        
        print(f"Request successful! Received data with lastUpdateId: {response.get('lastUpdateId')}")
        print(f"Number of bids: {len(response.get('bids', []))}")
        print(f"Number of asks: {len(response.get('asks', []))}")
        
        # Process data into a more usable format
        bids_df = pd.DataFrame(response["bids"], columns=["price", "quantity"], dtype=float)
        asks_df = pd.DataFrame(response["asks"], columns=["price", "quantity"], dtype=float)
        
        order_book = {
            "timestamp": datetime.now(),
            "last_update_id": response["lastUpdateId"],
            "bids": bids_df,
            "asks": asks_df
        }
        
        return order_book
    except Exception as e:
        print(f"Error fetching order book: {type(e).__name__} - {str(e)}")
        raise

# Helper function to actually run async code in Jupyter
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)"""),
    
    nbf.v4.new_code_cell("""# Fetch the order book data
order_book = run_async(get_binance_order_book("WLD-USDT"))

# Display a sample of the data
if order_book:
    print("\\nTop 5 bids:")
    print(order_book["bids"].head())
    
    print("\\nTop 5 asks:")
    print(order_book["asks"].head())
    
    # Calculate and display simple metrics
    best_bid = float(order_book["bids"]["price"].iloc[0])
    best_ask = float(order_book["asks"]["price"].iloc[0])
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_pct = (spread / mid_price) * 100
    
    print(f"\\nBest bid: {best_bid}")
    print(f"Best ask: {best_ask}")
    print(f"Mid price: {mid_price:.4f}")
    print(f"Spread: {spread:.6f} ({spread_pct:.4f}%)")
else:
    print("No order book data available")"""),
    
    nbf.v4.new_code_cell("""# Plot the order book
def plot_order_book(order_book, levels=10):
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
    total_bid_qty = bids["quantity"].sum()
    total_ask_qty = asks["quantity"].sum()
    imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    
    plt.figtext(0.5, 0.01, 
                f"Order Book Imbalance: {imbalance:.2f} ({'Buy pressure' if imbalance > 0 else 'Sell pressure'})\\n" 
                f"Total Bid Qty: {total_bid_qty:.2f} | Total Ask Qty: {total_ask_qty:.2f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

# Plot the order book if we have data
if order_book:
    plot_order_book(order_book)"""),
    
    nbf.v4.new_code_cell("""# Function to monitor the order book over time
def monitor_order_book(trading_pair, num_updates=5, interval_seconds=5):
    print(f"Starting order book monitoring for {trading_pair}")
    print(f"Will collect {num_updates} updates, {interval_seconds} seconds apart")
    
    # Initialize storage for the data
    timestamps = []
    mid_prices = []
    spreads = []
    imbalances = []
    best_bids = []
    best_asks = []
    
    # Run the monitoring loop
    for i in range(num_updates):
        print(f"\\nUpdate {i+1}/{num_updates}:")
        
        # Fetch the order book
        order_book = run_async(get_binance_order_book(trading_pair))
        
        if order_book:
            # Extract data
            timestamp = order_book["timestamp"]
            best_bid = float(order_book["bids"]["price"].iloc[0])
            best_ask = float(order_book["asks"]["price"].iloc[0])
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Calculate imbalance
            total_bid_qty = order_book["bids"]["quantity"].sum()
            total_ask_qty = order_book["asks"]["quantity"].sum()
            imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
            
            # Store the data
            timestamps.append(timestamp)
            mid_prices.append(mid_price)
            spreads.append(spread)
            imbalances.append(imbalance)
            best_bids.append(best_bid)
            best_asks.append(best_ask)
            
            # Print update
            print(f"Time: {timestamp.strftime('%H:%M:%S')}")
            print(f"Best bid: {best_bid}")
            print(f"Best ask: {best_ask}")
            print(f"Mid price: {mid_price:.4f}")
            print(f"Spread: {spread:.6f}")
            print(f"Imbalance: {imbalance:.4f}")
        else:
            print("Failed to fetch order book")
        
        # Wait for the next update (except for the last one)
        if i < num_updates - 1:
            print(f"Waiting {interval_seconds} seconds for next update...")
            import time
            time.sleep(interval_seconds)
    
    # Return the collected data
    return {
        "timestamps": timestamps,
        "mid_prices": mid_prices,
        "spreads": spreads,
        "imbalances": imbalances,
        "best_bids": best_bids,
        "best_asks": best_asks
    }

# Run the monitoring (uncomment to use)
# monitoring_data = monitor_order_book("WLD-USDT", num_updates=5, interval_seconds=10)"""),
    
    nbf.v4.new_code_cell("""# Plot the monitoring results
def plot_monitoring_results(data):
    if not data or len(data["timestamps"]) == 0:
        print("No monitoring data to plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Convert timestamps to strings for x-axis
    time_labels = [t.strftime('%H:%M:%S') for t in data["timestamps"]]
    x = range(len(time_labels))
    
    # Plot 1: Price chart with best bid and ask
    ax1.plot(x, data["best_bids"], 'g-', label='Best Bid')
    ax1.plot(x, data["best_asks"], 'r-', label='Best Ask')
    ax1.plot(x, data["mid_prices"], 'b--', label='Mid Price')
    ax1.set_title("WLD-USDT Price Evolution")
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Spread
    ax2.plot(x, data["spreads"], 'purple-', label='Spread')
    ax2.set_title("WLD-USDT Spread")
    ax2.set_ylabel("Spread")
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Imbalance
    ax3.bar(x, data["imbalances"], color=['green' if i > 0 else 'red' for i in data["imbalances"]], alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_title("WLD-USDT Order Book Imbalance")
    ax3.set_ylabel("Imbalance")
    ax3.set_xlabel("Time")
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_labels)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# Uncomment to plot monitoring results
# if 'monitoring_data' in locals():
#     plot_monitoring_results(monitoring_data)"""),
    
    nbf.v4.new_code_cell("""# Run the monitor function and plot results in one go
print("Starting order book monitoring...")
monitoring_data = monitor_order_book("WLD-USDT", num_updates=3, interval_seconds=5)

if monitoring_data and len(monitoring_data["timestamps"]) > 0:
    print("\\nPlotting monitoring results...")
    plot_monitoring_results(monitoring_data)
else:
    print("No monitoring data available to plot")""")
]

# Add cells to the notebook
nb['cells'] = cells

# Create the notebooks directory if it doesn't exist
os.makedirs('research_notebooks/order_book', exist_ok=True)

# Write the notebook to a file
with open('research_notebooks/order_book/wld_usdt_orderbook.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully at research_notebooks/order_book/wld_usdt_orderbook.ipynb") 