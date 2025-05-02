-- TFT Features Preparation Script - Simple Version
-- This creates and populates the TFT features table needed for model training

-- First, drop the table if it exists to avoid errors with primary key
DROP TABLE IF EXISTS tft_features;

-- 1. Create the TFT features table with timestamp as the primary key
CREATE TABLE tft_features (
    timestamp TIMESTAMPTZ NOT NULL,
    trading_pair TEXT NOT NULL,
    
    -- Market state features
    mid_price NUMERIC(18,8),
    spread NUMERIC(18,8),
    spread_pct NUMERIC(18,8),
    imbalance NUMERIC(18,8),
    
    -- Order flow features
    new_bid_orders INTEGER,
    new_ask_orders INTEGER,
    canceled_bid_orders INTEGER,
    canceled_ask_orders INTEGER,
    
    -- Target variables
    next_price_1min NUMERIC(18,8),
    next_price_5min NUMERIC(18,8),
    fill_probability NUMERIC(5,4),
    
    PRIMARY KEY (timestamp, trading_pair)
);

-- 2. Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('tft_features', 'timestamp');

-- 3. Insert sample data using a direct SQL approach
-- This avoids PL/pgSQL function issues
INSERT INTO tft_features
WITH 
-- Get time points at 1-minute intervals for the entire data period
time_points AS (
    SELECT generate_series(
        (SELECT date_trunc('minute', MIN(timestamp)) FROM order_book_summary),
        (SELECT date_trunc('minute', MAX(timestamp)) FROM order_events),
        '1 minute'::interval
    ) AS timestamp
),
-- Create features for each time point
features AS (
    SELECT
        tp.timestamp,
        'WLD-USDT' AS trading_pair,
        
        -- Get current market state
        (SELECT mid_price FROM order_book_summary 
         WHERE timestamp <= tp.timestamp
         ORDER BY timestamp DESC LIMIT 1) AS mid_price,
        
        (SELECT spread FROM order_book_summary 
         WHERE timestamp <= tp.timestamp
         ORDER BY timestamp DESC LIMIT 1) AS spread,
        
        (SELECT spread_pct FROM order_book_summary 
         WHERE timestamp <= tp.timestamp
         ORDER BY timestamp DESC LIMIT 1) AS spread_pct,
        
        (SELECT imbalance FROM order_book_summary 
         WHERE timestamp <= tp.timestamp
         ORDER BY timestamp DESC LIMIT 1) AS imbalance,
        
        -- Count orders in the last minute
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'NEW' 
         AND side = 'BID'
         AND timestamp BETWEEN tp.timestamp - INTERVAL '1 minute' AND tp.timestamp) AS new_bid_orders,
        
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'NEW' 
         AND side = 'ASK'
         AND timestamp BETWEEN tp.timestamp - INTERVAL '1 minute' AND tp.timestamp) AS new_ask_orders,
        
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'CANCEL' 
         AND side = 'BID'
         AND timestamp BETWEEN tp.timestamp - INTERVAL '1 minute' AND tp.timestamp) AS canceled_bid_orders,
        
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'CANCEL' 
         AND side = 'ASK'
         AND timestamp BETWEEN tp.timestamp - INTERVAL '1 minute' AND tp.timestamp) AS canceled_ask_orders,
        
        -- Get future prices (for targets)
        (SELECT mid_price FROM order_book_summary 
         WHERE timestamp >= tp.timestamp + INTERVAL '1 minute'
         ORDER BY timestamp ASC LIMIT 1) AS next_price_1min,
        
        (SELECT mid_price FROM order_book_summary 
         WHERE timestamp >= tp.timestamp + INTERVAL '5 minutes'
         ORDER BY timestamp ASC LIMIT 1) AS next_price_5min,
        
        -- Calculate fill probability based on order book imbalance and spread
        -- This is a simplified model - in reality you would use more sophisticated methods
        CASE 
            WHEN (SELECT imbalance FROM order_book_summary 
                 WHERE timestamp <= tp.timestamp
                 ORDER BY timestamp DESC LIMIT 1) > 0 THEN 0.5 + (SELECT imbalance FROM order_book_summary 
                                                               WHERE timestamp <= tp.timestamp
                                                               ORDER BY timestamp DESC LIMIT 1) * 0.5
            ELSE 0.5 - ABS((SELECT imbalance FROM order_book_summary 
                          WHERE timestamp <= tp.timestamp
                          ORDER BY timestamp DESC LIMIT 1)) * 0.5
        END AS fill_probability
    FROM time_points tp
)
SELECT * FROM features 
WHERE next_price_1min IS NOT NULL 
AND mid_price IS NOT NULL;

-- 4. Check results
SELECT COUNT(*) FROM tft_features;
SELECT MIN(timestamp), MAX(timestamp) FROM tft_features; 