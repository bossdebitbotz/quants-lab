-- Fix TimescaleDB integration issues
-- This is a simplified version focusing on core functionality

-- Check if TimescaleDB extension is loaded, if not, load it
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Create the table for detected fills first
CREATE TABLE IF NOT EXISTS detected_fills (
    id SERIAL PRIMARY KEY,
    synthetic_order_id VARCHAR(64) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    fill_timestamp TIMESTAMP NOT NULL,
    price NUMERIC(18,8) NOT NULL,
    quantity NUMERIC(18,8) NOT NULL,
    side VARCHAR(4) NOT NULL,
    detection_method VARCHAR(50) NOT NULL
);

-- Create index on detected_fills
CREATE INDEX IF NOT EXISTS detected_fills_order_id_idx 
ON detected_fills (synthetic_order_id);

CREATE INDEX IF NOT EXISTS detected_fills_timestamp_idx 
ON detected_fills (fill_timestamp);

-- 2. Detect potential fills (orders that disappeared without explicit cancellation)
INSERT INTO detected_fills (
    synthetic_order_id, trading_pair, fill_timestamp, price, quantity, side, detection_method
)
SELECT 
    ao.synthetic_order_id,
    ao.trading_pair,
    ao.last_updated as fill_timestamp,
    ao.price,
    ao.quantity,
    ao.side,
    'vanished_without_cancel' as detection_method
FROM active_orders ao
WHERE ao.status = 'ACTIVE' 
AND ao.synthetic_order_id NOT IN (
    SELECT synthetic_order_id 
    FROM order_events 
    WHERE event_type = 'CANCEL'
)
ON CONFLICT DO NOTHING;

-- Update the status in active_orders for these potential fills
UPDATE active_orders
SET status = 'FILLED'
WHERE status = 'ACTIVE'
AND synthetic_order_id IN (SELECT synthetic_order_id FROM detected_fills);

-- 3. Create time-bucketed features (1-minute intervals)
CREATE MATERIALIZED VIEW market_features_1min AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    trading_pair,
    AVG(mid_price) AS avg_mid_price,
    AVG(spread) AS avg_spread,
    AVG(spread_pct) AS avg_spread_pct,
    AVG(imbalance) AS avg_imbalance,
    MAX(best_bid) AS max_bid,
    MIN(best_ask) AS min_ask,
    SUM(bid_quantity) AS total_bid_qty,
    SUM(ask_quantity) AS total_ask_qty,
    COUNT(*) AS update_count
FROM order_book_summary
GROUP BY bucket, trading_pair;

-- 4. Create order event aggregations (event counts by type in time buckets)
CREATE MATERIALIZED VIEW order_events_1min AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    trading_pair,
    side,
    event_type,
    COUNT(*) AS event_count,
    AVG(price) AS avg_price,
    AVG(quantity) AS avg_quantity
FROM order_events
GROUP BY bucket, trading_pair, side, event_type;

-- 5. Create a view with order outcomes (for target variable creation)
CREATE OR REPLACE VIEW order_outcomes AS
WITH new_orders AS (
    SELECT 
        e.synthetic_order_id,
        e.trading_pair,
        e.timestamp AS creation_time,
        e.price,
        e.quantity,
        e.side
    FROM order_events e
    WHERE e.event_type = 'NEW'
),
cancel_events AS (
    SELECT 
        synthetic_order_id,
        timestamp AS cancel_time
    FROM order_events
    WHERE event_type = 'CANCEL'
),
fill_events AS (
    SELECT 
        synthetic_order_id,
        fill_timestamp
    FROM detected_fills
)
SELECT 
    no.synthetic_order_id,
    no.trading_pair,
    no.creation_time,
    no.price,
    no.quantity,
    no.side,
    ce.cancel_time,
    fe.fill_timestamp,
    CASE 
        WHEN fe.fill_timestamp IS NOT NULL THEN 'FILLED'
        WHEN ce.cancel_time IS NOT NULL THEN 'CANCELED'
        ELSE 'ACTIVE'
    END AS outcome,
    CASE 
        WHEN fe.fill_timestamp IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (fe.fill_timestamp - no.creation_time))
        WHEN ce.cancel_time IS NOT NULL THEN
            EXTRACT(EPOCH FROM (ce.cancel_time - no.creation_time))
        ELSE
            NULL
    END AS time_to_outcome_seconds
FROM new_orders no
LEFT JOIN cancel_events ce ON no.synthetic_order_id = ce.synthetic_order_id
LEFT JOIN fill_events fe ON no.synthetic_order_id = fe.synthetic_order_id;

-- 6. Create a TFT-ready features table with regular time intervals
CREATE TABLE IF NOT EXISTS tft_features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    
    -- Market state features (from order_book_summary)
    mid_price NUMERIC(18,8),
    spread NUMERIC(18,8),
    spread_pct NUMERIC(18,8),
    bid_ask_imbalance NUMERIC(18,8),
    
    -- Order flow features (from order_events aggregations)
    new_bid_orders_1min INTEGER,
    new_ask_orders_1min INTEGER,
    canceled_bid_orders_1min INTEGER,
    canceled_ask_orders_1min INTEGER,
    
    -- Target variables (to predict)
    fill_probability_bid NUMERIC(5,4),
    fill_probability_ask NUMERIC(5,4),
    avg_time_to_fill NUMERIC(10,4)
);

-- Create index on time and pair
CREATE INDEX IF NOT EXISTS tft_features_time_pair_idx 
ON tft_features (timestamp, trading_pair);

-- Convert to hypertable - do this before inserting data
SELECT create_hypertable('tft_features', 'timestamp', if_not_exists => TRUE, 
                        create_default_indexes => TRUE);

-- 7. Insert some sample prepared TFT data from the most recent time period
INSERT INTO tft_features (
    timestamp,
    trading_pair,
    mid_price,
    spread,
    spread_pct,
    bid_ask_imbalance,
    new_bid_orders_1min,
    new_ask_orders_1min,
    canceled_bid_orders_1min,
    canceled_ask_orders_1min,
    fill_probability_bid,
    fill_probability_ask,
    avg_time_to_fill
)
WITH 
-- Get unique timestamps at 1-minute intervals
time_buckets AS (
    SELECT DISTINCT time_bucket('1 minute', timestamp) AS bucket_time
    FROM order_book_summary
    ORDER BY bucket_time
    LIMIT 60 -- Get just the first 60 sample points
),
-- For each timestamp, calculate features
features AS (
    SELECT
        tb.bucket_time AS timestamp,
        'WLD-USDT' AS trading_pair,
        
        -- Market state (from closest order_book_summary)
        (SELECT mid_price FROM order_book_summary 
         WHERE timestamp <= tb.bucket_time
         ORDER BY timestamp DESC LIMIT 1) AS mid_price,
         
        (SELECT spread FROM order_book_summary 
         WHERE timestamp <= tb.bucket_time
         ORDER BY timestamp DESC LIMIT 1) AS spread,
         
        (SELECT spread_pct FROM order_book_summary 
         WHERE timestamp <= tb.bucket_time
         ORDER BY timestamp DESC LIMIT 1) AS spread_pct,
         
        (SELECT imbalance FROM order_book_summary 
         WHERE timestamp <= tb.bucket_time
         ORDER BY timestamp DESC LIMIT 1) AS bid_ask_imbalance,
        
        -- Order flow counts in the past minute
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'NEW' AND side = 'BID'
         AND timestamp BETWEEN tb.bucket_time - INTERVAL '1 minute' AND tb.bucket_time) AS new_bid_orders,
         
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'NEW' AND side = 'ASK'
         AND timestamp BETWEEN tb.bucket_time - INTERVAL '1 minute' AND tb.bucket_time) AS new_ask_orders,
         
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'CANCEL' AND side = 'BID'
         AND timestamp BETWEEN tb.bucket_time - INTERVAL '1 minute' AND tb.bucket_time) AS canceled_bid_orders,
         
        (SELECT COUNT(*) FROM order_events 
         WHERE event_type = 'CANCEL' AND side = 'ASK'
         AND timestamp BETWEEN tb.bucket_time - INTERVAL '1 minute' AND tb.bucket_time) AS canceled_ask_orders,
         
        -- Fill probabilities and timing (placeholder values for now)
        0.1 AS fill_probability_bid,
        0.1 AS fill_probability_ask,
        10.0 AS avg_time_to_fill
        
    FROM time_buckets tb
)
SELECT * FROM features;

-- 8. Create hypertables from existing tables (may need adjustment if tables are large)
-- Try to convert order_events to a hypertable if it isn't already
SELECT create_hypertable('order_events', 'timestamp', if_not_exists => TRUE,
                         create_default_indexes => FALSE);
                         
-- Try to convert order_book_summary to a hypertable if it isn't already
SELECT create_hypertable('order_book_summary', 'timestamp', if_not_exists => TRUE,
                         create_default_indexes => FALSE); 