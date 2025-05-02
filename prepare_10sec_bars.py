import pandas as pd
import numpy as np
import psycopg2
import os
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("10sec_bar_preparation")

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

def fetch_order_book_data(conn, trading_pair="WLD-USDT", start_date=None, end_date=None, limit=None):
    """
    Fetch order book data from the database
    
    Args:
        conn: Database connection
        trading_pair: Trading pair to fetch data for
        start_date: Start date string (format: 'YYYY-MM-DD')
        end_date: End date string (format: 'YYYY-MM-DD')
        limit: Optional limit on number of rows to fetch
    
    Returns:
        DataFrame with order book data
    """
    where_clause = "WHERE trading_pair = %s"
    params = [trading_pair]
    
    if start_date:
        where_clause += " AND timestamp >= %s"
        params.append(start_date)
        
        if end_date:
            where_clause += " AND timestamp <= %s"
            params.append(end_date)
    elif end_date:
        where_clause += " AND timestamp <= %s"
        params.append(end_date)
    
    limit_clause = f" LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        timestamp,
        price,
        quantity,
        order_type
    FROM order_book
    {where_clause}
    ORDER BY timestamp
    {limit_clause}
    """
    
    logger.info(f"Fetching order book data with query: {query}")
    df = pd.read_sql_query(query, conn, params=params)
    
    # Pivot to get bid and ask prices/sizes
    bids = df[df['order_type'] == 'bid'].copy()
    asks = df[df['order_type'] == 'ask'].copy()
    
    # Group by timestamp and get the best bid/ask
    bids = bids.sort_values(['timestamp', 'price'], ascending=[True, False]).groupby('timestamp').first().reset_index()
    asks = asks.sort_values(['timestamp', 'price'], ascending=[True, True]).groupby('timestamp').first().reset_index()
    
    # Rename columns
    bids.rename(columns={'price': 'bid_price', 'quantity': 'bid_size'}, inplace=True)
    asks.rename(columns={'price': 'ask_price', 'quantity': 'ask_size'}, inplace=True)
    
    # Join on timestamp
    result = pd.merge(bids[['timestamp', 'bid_price', 'bid_size']], 
                      asks[['timestamp', 'ask_price', 'ask_size']], 
                      on='timestamp', how='outer')
    
    logger.info(f"Fetched {len(result)} order book records")
    return result

def fetch_trades_data(conn, trading_pair="WLD-USDT", start_date=None, end_date=None, limit=None):
    """
    Fetch trades data from the database
    
    Args:
        conn: Database connection
        trading_pair: Trading pair to fetch data for
        start_date: Start date string (format: 'YYYY-MM-DD')
        end_date: End date string (format: 'YYYY-MM-DD')
        limit: Optional limit on number of rows to fetch
    
    Returns:
        DataFrame with trades data
    """
    where_clause = "WHERE trading_pair = %s"
    params = [trading_pair]
    
    if start_date:
        where_clause += " AND timestamp >= %s"
        params.append(start_date)
        
        if end_date:
            where_clause += " AND timestamp <= %s"
            params.append(end_date)
    elif end_date:
        where_clause += " AND timestamp <= %s"
        params.append(end_date)
    
    limit_clause = f" LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        timestamp,
        price as trade_price,
        quantity as trade_size,
        side as trade_side
    FROM trades
    {where_clause}
    ORDER BY timestamp
    {limit_clause}
    """
    
    logger.info(f"Fetching trades data with query: {query}")
    df = pd.read_sql_query(query, conn, params=params)
    logger.info(f"Fetched {len(df)} trades records")
    return df

def fetch_order_events_data(conn, trading_pair="WLD-USDT", start_date=None, end_date=None, limit=None):
    """
    Fetch order events data from the database
    
    Args:
        conn: Database connection
        trading_pair: Trading pair to fetch data for
        start_date: Start date string (format: 'YYYY-MM-DD')
        end_date: End date string (format: 'YYYY-MM-DD')
        limit: Optional limit on number of rows to fetch
    
    Returns:
        DataFrame with order events data
    """
    where_clause = "WHERE trading_pair = %s"
    params = [trading_pair]
    
    if start_date:
        where_clause += " AND timestamp >= %s"
        params.append(start_date)
        
        if end_date:
            where_clause += " AND timestamp <= %s"
            params.append(end_date)
    elif end_date:
        where_clause += " AND timestamp <= %s"
        params.append(end_date)
    
    limit_clause = f" LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        timestamp,
        synthetic_order_id as order_id,
        event_type as order_status,
        side as order_type,
        price,
        quantity
    FROM order_events
    {where_clause}
    ORDER BY timestamp
    {limit_clause}
    """
    
    logger.info(f"Fetching order events data with query: {query}")
    df = pd.read_sql_query(query, conn, params=params)
    
    # Map event types to simplified order status
    status_map = {
        'NEW': 'new',
        'MODIFY': 'modified',
        'CANCEL': 'canceled',
        'FILL': 'executed'
    }
    df['order_status'] = df['order_status'].map(status_map)
    
    logger.info(f"Fetched {len(df)} order events records")
    return df

def merge_raw_data(order_book_data, trades_data, order_events_data):
    """
    Merge all data sources into a single DataFrame with aligned timestamps
    
    Args:
        order_book_data: DataFrame with order book data
        trades_data: DataFrame with trades data
        order_events_data: DataFrame with order events data
    
    Returns:
        Merged DataFrame with all data
    """
    # Create a common timestamp index across all datasets
    all_timestamps = pd.concat([
        order_book_data['timestamp'],
        trades_data['timestamp'],
        order_events_data['timestamp']
    ]).drop_duplicates().sort_values().reset_index(drop=True)
    
    # Create a new DataFrame with all timestamps
    merged_df = pd.DataFrame({'timestamp': all_timestamps})
    
    # Merge with order book data
    merged_df = pd.merge(merged_df, order_book_data, 
                         on='timestamp', how='left')
    
    # Merge with trades data
    merged_df = pd.merge(merged_df, trades_data, 
                         on='timestamp', how='left')
    
    # Merge with order events data
    merged_df = pd.merge(merged_df, order_events_data, 
                         suffixes=('', '_events'),
                         on='timestamp', how='left')
    
    return merged_df

def create_10sec_bars(raw_data):
    """
    Convert raw data to 10-second OHLCV bars
    
    Args:
        raw_data: Merged DataFrame with order book, trades, and order events data
    
    Returns:
        DataFrame with 10-second bars
    """
    # Ensure timestamp is parsed as datetime
    if raw_data.empty:
        logger.warning("No raw data available to create bars")
        return pd.DataFrame()
    
    # Calculate mid price
    raw_data['mid_price'] = (raw_data['bid_price'].fillna(0) + raw_data['ask_price'].fillna(0)) / 2
    raw_data['spread'] = raw_data['ask_price'].fillna(0) - raw_data['bid_price'].fillna(0)
    
    # Handle case where mid_price is 0
    mask = raw_data['mid_price'] != 0
    raw_data['spread_pct'] = np.where(mask, raw_data['spread'] / raw_data['mid_price'], 0)
    
    # Set timestamp as index for resampling
    raw_data.set_index('timestamp', inplace=True)
    
    # Resample to 10-second bars
    bars = pd.DataFrame()
    
    # Price and volume data
    try:
        ohlc = raw_data['trade_price'].resample('10S').ohlc()
        bars['open'] = ohlc['open']
        bars['high'] = ohlc['high']
        bars['low'] = ohlc['low']
        bars['close'] = ohlc['close']
    except Exception as e:
        logger.error(f"Error creating OHLC: {e}")
        # Use mid price if trade price is not available
        ohlc = raw_data['mid_price'].resample('10S').ohlc()
        bars['open'] = ohlc['open']
        bars['high'] = ohlc['high']
        bars['low'] = ohlc['low']
        bars['close'] = ohlc['close']
    
    # Calculate volume from trades
    volume = raw_data[raw_data['trade_size'].notnull()]
    bars['volume'] = volume['trade_size'].resample('10S').sum()
    
    # Order book metrics
    bars['bid_vol'] = raw_data['bid_size'].resample('10S').mean()
    bars['ask_vol'] = raw_data['ask_size'].resample('10S').mean()
    
    # Calculate imbalance with handling for NaN and 0 sum values
    bid_vol = raw_data['bid_size'].resample('10S').mean()
    ask_vol = raw_data['ask_size'].resample('10S').mean()
    sum_vol = bid_vol + ask_vol
    bars['imbalance'] = np.where(sum_vol != 0, (bid_vol - ask_vol) / sum_vol, 0)
    
    # Spread metrics
    bars['spread_mean'] = raw_data['spread'].resample('10S').mean()
    bars['spread_min'] = raw_data['spread'].resample('10S').min()
    bars['spread_max'] = raw_data['spread'].resample('10S').max()
    bars['spread_pct_mean'] = raw_data['spread_pct'].resample('10S').mean()
    
    # Calculate order flow
    order_data = raw_data[raw_data['order_id'].notnull()]
    
    # New orders
    new_orders = order_data[order_data['order_status'] == 'new']
    bars['new_bid_orders'] = new_orders[new_orders['order_type'] == 'BID']['order_id'].resample('10S').count()
    bars['new_ask_orders'] = new_orders[new_orders['order_type'] == 'ASK']['order_id'].resample('10S').count()
    
    # Canceled orders
    canceled_orders = order_data[order_data['order_status'] == 'canceled']
    bars['canceled_bid_orders'] = canceled_orders[canceled_orders['order_type'] == 'BID']['order_id'].resample('10S').count()
    bars['canceled_ask_orders'] = canceled_orders[canceled_orders['order_type'] == 'ASK']['order_id'].resample('10S').count()
    
    # Executed orders
    executed_orders = order_data[order_data['order_status'] == 'executed']
    bars['executed_bid_orders'] = executed_orders[executed_orders['order_type'] == 'BID']['order_id'].resample('10S').count()
    bars['executed_ask_orders'] = executed_orders[executed_orders['order_type'] == 'ASK']['order_id'].resample('10S').count()
    
    # Calculate trade direction
    trades = raw_data[raw_data['trade_side'].notnull()]
    buy_volume = trades[trades['trade_side'] == 'buy']['trade_size'].resample('10S').sum()
    sell_volume = trades[trades['trade_side'] == 'sell']['trade_size'].resample('10S').sum()
    
    # Buy-sell imbalance
    total_volume = buy_volume.add(sell_volume, fill_value=0)
    bars['buy_sell_imbalance'] = buy_volume.sub(sell_volume, fill_value=0) / total_volume.replace(0, np.nan)
    
    # Price momentum features
    bars['returns_10sec'] = bars['close'].pct_change()
    bars['returns_30sec'] = bars['close'].pct_change(3)  # 3 * 10sec = 30sec
    bars['returns_1min'] = bars['close'].pct_change(6)   # 6 * 10sec = 1min
    
    # Rolling volatility
    bars['volatility_1min'] = bars['returns_10sec'].rolling(6).std()
    
    # Target: 10-second forward returns
    bars['target_10sec'] = bars['close'].pct_change().shift(-1)
    bars['target_30sec'] = bars['close'].pct_change(3).shift(-3)
    bars['target_1min'] = bars['close'].pct_change(6).shift(-6)
    
    # Fill missing values with 0 (for count-based features) or forward fill (for others)
    count_cols = ['new_bid_orders', 'new_ask_orders', 'canceled_bid_orders', 
                 'canceled_ask_orders', 'executed_bid_orders', 'executed_ask_orders']
    
    for col in count_cols:
        bars[col] = bars[col].fillna(0)
    
    # Forward fill price data
    price_cols = ['open', 'high', 'low', 'close']
    bars[price_cols] = bars[price_cols].ffill()
    
    # Fill remaining NaNs with column means
    bars = bars.fillna(bars.mean())
    
    # Drop any remaining NaN rows
    bars.dropna(inplace=True)
    
    logger.info(f"Created {len(bars)} 10-second bars")
    return bars

def store_10sec_bars(bars, conn, trading_pair="WLD-USDT"):
    """
    Store 10-second bars in the database
    
    Args:
        bars: DataFrame with 10-second bars
        conn: Database connection
        trading_pair: Trading pair for the data
    """
    if bars.empty:
        logger.warning("No bars to store")
        return
    
    # Reset index to convert timestamp to column
    bars_to_store = bars.reset_index()
    
    # Add trading pair column
    bars_to_store['trading_pair'] = trading_pair
    
    # Map columns to the expected schema
    column_mapping = {
        'mid_price': 'close',  # Use mid_price as close if close doesn't exist
        'spread_pct': 'spread_pct_mean',  # Map spread_pct to spread_pct_mean
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in bars_to_store.columns and new_col not in bars_to_store.columns:
            bars_to_store[new_col] = bars_to_store[old_col]
    
    # Ensure we have all required columns or set them to NULL
    required_columns = [
        'timestamp', 'trading_pair', 'open', 'high', 'low', 'close',
        'volume', 'bid_vol', 'ask_vol', 'imbalance', 'spread_mean',
        'spread_min', 'spread_max', 'spread_pct_mean', 'new_bid_orders',
        'new_ask_orders', 'canceled_bid_orders', 'canceled_ask_orders',
        'executed_bid_orders', 'executed_ask_orders', 'buy_sell_imbalance',
        'returns_10sec', 'returns_30sec', 'returns_1min', 'volatility_1min',
        'target_10sec', 'target_30sec', 'target_1min'
    ]
    
    for col in required_columns:
        if col not in bars_to_store.columns:
            if col in ['timestamp', 'trading_pair']:
                logger.error(f"Required column {col} missing")
                return
            else:
                logger.warning(f"Column {col} missing, setting to NULL")
                bars_to_store[col] = None
    
    # Create table if it doesn't exist
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS tft_features_10sec (
        timestamp TIMESTAMPTZ PRIMARY KEY,
        trading_pair VARCHAR(20) NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume FLOAT,
        bid_vol FLOAT,
        ask_vol FLOAT,
        imbalance FLOAT,
        spread_mean FLOAT,
        spread_min FLOAT,
        spread_max FLOAT,
        spread_pct_mean FLOAT,
        new_bid_orders INT,
        new_ask_orders INT,
        canceled_bid_orders INT,
        canceled_ask_orders INT,
        executed_bid_orders INT,
        executed_ask_orders INT,
        buy_sell_imbalance FLOAT,
        returns_10sec FLOAT,
        returns_30sec FLOAT,
        returns_1min FLOAT,
        volatility_1min FLOAT,
        target_10sec FLOAT,
        target_30sec FLOAT,
        target_1min FLOAT
    )
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    # Only include columns that exist in the table
    column_subset = [col for col in required_columns if col in bars_to_store.columns]
    insert_data = bars_to_store[column_subset]
    
    # Insert data
    for _, row in insert_data.iterrows():
        columns = ', '.join(row.index)
        placeholders = ', '.join(['%s'] * len(row))
        
        insert_query = f"""
        INSERT INTO tft_features_10sec ({columns})
        VALUES ({placeholders})
        ON CONFLICT (timestamp) DO UPDATE SET
        """
        
        # Create SET clause dynamically
        set_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in row.index if col != 'timestamp'])
        insert_query += set_clause
        
        cursor.execute(insert_query, tuple(row))
    
    conn.commit()
    logger.info(f"Stored {len(insert_data)} 10-second bars in the database")

def fetch_existing_features(conn, start_date=None, end_date=None):
    """
    Check if we already have data in tft_features that can be resampled to 10 seconds
    
    Args:
        conn: Database connection
        start_date: Start date string (format: 'YYYY-MM-DD')
        end_date: End date string (format: 'YYYY-MM-DD')
        
    Returns:
        DataFrame with features if available, None otherwise
    """
    where_clause = ""
    params = []
    
    if start_date:
        where_clause += " WHERE timestamp >= %s"
        params.append(start_date)
        
        if end_date:
            where_clause += " AND timestamp <= %s"
            params.append(end_date)
    elif end_date:
        where_clause += " WHERE timestamp <= %s"
        params.append(end_date)
    
    query = f"""
    SELECT 
        timestamp,
        trading_pair,
        mid_price,
        spread,
        spread_pct,
        imbalance,
        new_bid_orders,
        new_ask_orders,
        canceled_bid_orders,
        canceled_ask_orders,
        fill_probability
    FROM tft_features
    {where_clause}
    ORDER BY timestamp
    """
    
    try:
        logger.info(f"Checking for existing features with query: {query}")
        df = pd.read_sql_query(query, conn, params=params)
        
        # Ensure numeric columns are numeric
        numeric_columns = ['mid_price', 'spread', 'spread_pct', 'imbalance', 
                          'new_bid_orders', 'new_ask_orders', 
                          'canceled_bid_orders', 'canceled_ask_orders',
                          'fill_probability']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if not df.empty:
            logger.info(f"Found {len(df)} existing feature records")
            return df
        else:
            logger.info("No existing features found")
            return None
    except Exception as e:
        logger.warning(f"Error checking existing features: {e}")
        return None

def main():
    logger.info("Starting 10-second bar preparation")
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to the database")
        return
    
    try:
        # Fetch data for the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Check if we can use existing features
        existing_features = fetch_existing_features(conn, start_date_str, end_date_str)
        
        if existing_features is not None and not existing_features.empty:
            logger.info("Using existing features as foundation")
            
            # Set timestamp as index and prepare for resampling
            existing_features.set_index('timestamp', inplace=True)
            
            # Log feature columns
            logger.info(f"Available feature columns: {existing_features.columns.tolist()}")
            
            # Get only the numeric columns for resampling
            numeric_features = existing_features.select_dtypes(include=['number'])
            logger.info(f"Numeric feature columns: {numeric_features.columns.tolist()}")
            
            # Resample numeric features to 10-second bars
            resampled = numeric_features.resample('10S').mean()
            logger.info(f"Resampled to {len(resampled)} 10-second intervals")
            
            # Check for NaN values before calculations
            nan_counts = resampled.isna().sum()
            logger.info(f"NaN counts before calculations: {nan_counts.to_dict()}")
            
            # Fill NaN values with forward fill first
            resampled = resampled.ffill()
            
            # Add missing features for 10-second bars
            if 'mid_price' in resampled.columns:
                resampled['returns_10sec'] = resampled['mid_price'].pct_change(fill_method=None)
                resampled['returns_30sec'] = resampled['mid_price'].pct_change(3, fill_method=None)
                resampled['returns_1min'] = resampled['mid_price'].pct_change(6, fill_method=None)
                resampled['volatility_1min'] = resampled['returns_10sec'].rolling(6).std()
                resampled['target_10sec'] = resampled['mid_price'].pct_change(fill_method=None).shift(-1)
                resampled['target_30sec'] = resampled['mid_price'].pct_change(3, fill_method=None).shift(-3)
                resampled['target_1min'] = resampled['mid_price'].pct_change(6, fill_method=None).shift(-6)
                
                # Map column names to match the target table schema
                resampled['close'] = resampled['mid_price']
            else:
                logger.error("'mid_price' column not found in features")
            
            if 'spread_pct' in resampled.columns:
                resampled['spread_pct_mean'] = resampled['spread_pct']
            
            # Check for NaN values after calculations
            nan_counts = resampled.isna().sum()
            logger.info(f"NaN counts after calculations: {nan_counts.to_dict()}")
            
            # Fill NaN values with backfill for remaining NaNs
            resampled = resampled.bfill()
            
            # Fill any remaining NaNs with 0
            resampled = resampled.fillna(0)
            
            # Create placeholder values for required columns
            if 'open' not in resampled.columns and 'close' in resampled.columns:
                resampled['open'] = resampled['close']
            if 'high' not in resampled.columns and 'close' in resampled.columns:
                resampled['high'] = resampled['close']
            if 'low' not in resampled.columns and 'close' in resampled.columns:
                resampled['low'] = resampled['close']
            
            # Skip the first and last few rows which may have NaN values due to rolling/shifting
            skip_rows = 10  # Skip first few rows which might have NaNs from lookback
            if len(resampled) > skip_rows * 2:
                resampled = resampled.iloc[skip_rows:-skip_rows]
                
            logger.info(f"Final resampled data shape: {resampled.shape}")
            
            if not resampled.empty:
                # Store the resampled bars
                store_10sec_bars(resampled, conn)
            else:
                logger.warning("No data to store after processing")
            
        else:
            logger.info("Fetching raw data to create 10-second bars")
            
            # Fetch raw data from different tables
            trading_pair = "WLD-USDT"
            
            order_book_data = fetch_order_book_data(
                conn, 
                trading_pair=trading_pair,
                start_date=start_date_str, 
                end_date=end_date_str
            )
            
            trades_data = fetch_trades_data(
                conn, 
                trading_pair=trading_pair,
                start_date=start_date_str, 
                end_date=end_date_str
            )
            
            order_events_data = fetch_order_events_data(
                conn, 
                trading_pair=trading_pair,
                start_date=start_date_str, 
                end_date=end_date_str
            )
            
            # Merge all data
            raw_data = merge_raw_data(order_book_data, trades_data, order_events_data)
            
            # Create 10-second bars
            bars = create_10sec_bars(raw_data)
            
            # Store bars in the database
            store_10sec_bars(bars, conn, trading_pair)
        
        logger.info("10-second bar preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in 10-second bar preparation: {str(e)}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main() 