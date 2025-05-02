import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("preprocessor")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

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
        logger.error(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def calculate_order_book_imbalance(bids, asks):
    """Calculate order book imbalance"""
    bid_volume = bids['quantity'].sum()
    ask_volume = asks['quantity'].sum()
    total_volume = bid_volume + ask_volume
    return (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

def calculate_price_impact(orders, side):
    """Calculate price impact for a given side"""
    if orders.empty:
        return 0
    
    # Sort by price (ascending for asks, descending for bids)
    orders = orders.sort_values('price', ascending=(side == 'ASK'))
    
    # Calculate cumulative volume
    orders['cum_volume'] = orders['quantity'].cumsum()
    
    # Calculate weighted average price
    orders['weighted_price'] = orders['price'] * orders['quantity']
    return orders['weighted_price'].sum() / orders['quantity'].sum()

def calculate_volume_profile(orders, num_levels=10):
    """Calculate volume profile for top N levels"""
    if orders.empty:
        return pd.Series([0] * num_levels)
    
    # Sort by price and take top N levels
    orders = orders.sort_values('price', ascending=False).head(num_levels)
    return orders['quantity'].values

def aggregate_order_events():
    """Aggregate order events into 1-minute intervals and calculate features"""
    try:
        conn = get_db_connection()
        
        # Query to get all order events
        query = """
        SELECT 
            timestamp,
            event_type,
            price,
            quantity,
            side,
            remaining_quantity
        FROM order_events
        ORDER BY timestamp
        """
        
        # Read data into DataFrame
        df = pd.read_sql_query(query, conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create 1-minute intervals
        df['minute'] = df['timestamp'].dt.floor('1min')
        
        # Initialize feature storage
        features = []
        
        # Process each 1-minute interval
        for minute, group in df.groupby('minute'):
            # Split into bids and asks
            bids = group[group['side'] == 'BID']
            asks = group[group['side'] == 'ASK']
            
            # Calculate features
            features.append({
                'timestamp': minute,
                'order_book_imbalance': calculate_order_book_imbalance(bids, asks),
                'bid_price_impact': calculate_price_impact(bids, 'BID'),
                'ask_price_impact': calculate_price_impact(asks, 'ASK'),
                'bid_volume': bids['quantity'].sum(),
                'ask_volume': asks['quantity'].sum(),
                'new_orders': len(group[group['event_type'] == 'NEW']),
                'cancelled_orders': len(group[group['event_type'] == 'CANCEL']),
                'modified_orders': len(group[group['event_type'] == 'MODIFY']),
                'filled_orders': len(group[group['event_type'] == 'FILL']),
                'bid_volume_profile': calculate_volume_profile(bids),
                'ask_volume_profile': calculate_volume_profile(asks)
            })
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Save to database
        with conn.cursor() as cursor:
            # Create features table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tft_features (
                    timestamp TIMESTAMP PRIMARY KEY,
                    order_book_imbalance DECIMAL(18, 8),
                    bid_price_impact DECIMAL(18, 8),
                    ask_price_impact DECIMAL(18, 8),
                    bid_volume DECIMAL(18, 8),
                    ask_volume DECIMAL(18, 8),
                    new_orders INTEGER,
                    cancelled_orders INTEGER,
                    modified_orders INTEGER,
                    filled_orders INTEGER,
                    bid_volume_profile DECIMAL(18, 8)[],
                    ask_volume_profile DECIMAL(18, 8)[]
                );
            """)
            
            # Insert features
            for _, row in features_df.iterrows():
                cursor.execute("""
                    INSERT INTO tft_features 
                    (timestamp, order_book_imbalance, bid_price_impact, ask_price_impact,
                     bid_volume, ask_volume, new_orders, cancelled_orders, modified_orders,
                     filled_orders, bid_volume_profile, ask_volume_profile)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO UPDATE
                    SET order_book_imbalance = EXCLUDED.order_book_imbalance,
                        bid_price_impact = EXCLUDED.bid_price_impact,
                        ask_price_impact = EXCLUDED.ask_price_impact,
                        bid_volume = EXCLUDED.bid_volume,
                        ask_volume = EXCLUDED.ask_volume,
                        new_orders = EXCLUDED.new_orders,
                        cancelled_orders = EXCLUDED.cancelled_orders,
                        modified_orders = EXCLUDED.modified_orders,
                        filled_orders = EXCLUDED.filled_orders,
                        bid_volume_profile = EXCLUDED.bid_volume_profile,
                        ask_volume_profile = EXCLUDED.ask_volume_profile;
                """, (
                    row['timestamp'],
                    row['order_book_imbalance'],
                    row['bid_price_impact'],
                    row['ask_price_impact'],
                    row['bid_volume'],
                    row['ask_volume'],
                    row['new_orders'],
                    row['cancelled_orders'],
                    row['modified_orders'],
                    row['filled_orders'],
                    row['bid_volume_profile'].tolist(),
                    row['ask_volume_profile'].tolist()
                ))
            
            conn.commit()
            logger.info(f"Processed and saved {len(features_df)} minutes of features")
            
    except Exception as e:
        logger.error(f"Error in feature aggregation: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logger.info("Starting order event aggregation and feature calculation")
    aggregate_order_events()
    logger.info("Feature calculation completed") 