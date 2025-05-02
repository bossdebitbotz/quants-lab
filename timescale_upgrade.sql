-- TimescaleDB Optimization Script for TFT Training
-- This script implements modifications to make the database more suitable for
-- Temporal Fusion Transformer model training

-- 1. Convert existing tables to TimescaleDB hypertables
-- This enables time-based partitioning and optimization
SELECT create_hypertable('order_events', 'timestamp', if_not_exists => TRUE, 
                          create_default_indexes => FALSE);
SELECT create_hypertable('order_book_summary', 'timestamp', if_not_exists => TRUE,
                          create_default_indexes => FALSE);
SELECT create_hypertable('order_book', 'timestamp', if_not_exists => TRUE,
                          create_default_indexes => FALSE);

-- 2. Create a table to store potential fill events
-- Since we don't have explicit FILL events, we'll detect them
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

-- 3. Detect potential fills (orders that disappeared without explicit cancellation)
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

-- 4. Create time-bucketed features (1-minute intervals)
CREATE MATERIALIZED VIEW IF NOT EXISTS market_features_1min AS
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

-- 5. Create 5-minute market features for longer horizons
CREATE MATERIALIZED VIEW IF NOT EXISTS market_features_5min AS
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
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

-- 6. Create order event aggregations (event counts by type in time buckets)
CREATE MATERIALIZED VIEW IF NOT EXISTS order_events_1min AS
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

-- 7. Create a view with order outcomes (for target variable creation)
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

-- 8. Create a TFT-ready features table with regular time intervals
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
    
    -- Price movement features
    price_change_1min NUMERIC(18,8),
    price_change_5min NUMERIC(18,8),
    
    -- Volatility features
    price_volatility_1min NUMERIC(18,8),
    
    -- Target variables (to predict)
    next_mid_price_1min NUMERIC(18,8),
    next_mid_price_5min NUMERIC(18,8),
    fill_probability_bid NUMERIC(5,4),
    fill_probability_ask NUMERIC(5,4),
    avg_time_to_fill NUMERIC(10,4)
);

-- Create index on time and pair
CREATE INDEX IF NOT EXISTS tft_features_time_pair_idx 
ON tft_features (timestamp, trading_pair);

-- Convert to hypertable
SELECT create_hypertable('tft_features', 'timestamp', if_not_exists => TRUE);

-- 9. Create a procedure to populate the TFT features at regular intervals
CREATE OR REPLACE PROCEDURE populate_tft_features(
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    interval_minutes INTEGER DEFAULT 1
)
LANGUAGE plpgsql
AS $$
DECLARE
    current_time TIMESTAMP;
BEGIN
    current_time := start_time;
    
    WHILE current_time <= end_time LOOP
        -- Insert a new record with features calculated at this timestamp
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
            price_change_1min,
            price_change_5min,
            price_volatility_1min,
            next_mid_price_1min,
            next_mid_price_5min,
            fill_probability_bid,
            fill_probability_ask,
            avg_time_to_fill
        )
        WITH 
        -- Get the current market state
        current_market AS (
            SELECT 
                'WLD-USDT' AS trading_pair,
                mid_price,
                spread,
                spread_pct,
                imbalance AS bid_ask_imbalance
            FROM order_book_summary
            WHERE trading_pair = 'WLD-USDT'
            AND timestamp <= current_time
            ORDER BY timestamp DESC
            LIMIT 1
        ),
        -- Calculate order flow metrics for the past minute
        order_flow AS (
            SELECT
                COUNT(*) FILTER (WHERE event_type = 'NEW' AND side = 'BID') AS new_bid_orders,
                COUNT(*) FILTER (WHERE event_type = 'NEW' AND side = 'ASK') AS new_ask_orders,
                COUNT(*) FILTER (WHERE event_type = 'CANCEL' AND side = 'BID') AS canceled_bid_orders,
                COUNT(*) FILTER (WHERE event_type = 'CANCEL' AND side = 'ASK') AS canceled_ask_orders
            FROM order_events
            WHERE trading_pair = 'WLD-USDT'
            AND timestamp BETWEEN current_time - INTERVAL '1 minute' AND current_time
        ),
        -- Calculate price changes
        price_changes AS (
            SELECT
                current_market.mid_price - prev_1min.mid_price AS price_change_1min,
                current_market.mid_price - prev_5min.mid_price AS price_change_5min,
                STDDEV(obs.mid_price) AS price_volatility_1min,
                next_1min.mid_price AS next_mid_price_1min,
                next_5min.mid_price AS next_mid_price_5min
            FROM current_market
            CROSS JOIN LATERAL (
                SELECT mid_price 
                FROM order_book_summary 
                WHERE trading_pair = 'WLD-USDT'
                AND timestamp <= current_time - INTERVAL '1 minute'
                ORDER BY timestamp DESC LIMIT 1
            ) prev_1min
            CROSS JOIN LATERAL (
                SELECT mid_price
                FROM order_book_summary
                WHERE trading_pair = 'WLD-USDT'
                AND timestamp <= current_time - INTERVAL '5 minutes'
                ORDER BY timestamp DESC LIMIT 1
            ) prev_5min
            CROSS JOIN LATERAL (
                SELECT AVG(mid_price) AS mid_price
                FROM order_book_summary
                WHERE trading_pair = 'WLD-USDT'
                AND timestamp BETWEEN current_time - INTERVAL '1 minute' AND current_time
            ) obs
            CROSS JOIN LATERAL (
                SELECT mid_price
                FROM order_book_summary
                WHERE trading_pair = 'WLD-USDT'
                AND timestamp >= current_time + INTERVAL '1 minute'
                ORDER BY timestamp ASC LIMIT 1
            ) next_1min
            CROSS JOIN LATERAL (
                SELECT mid_price
                FROM order_book_summary
                WHERE trading_pair = 'WLD-USDT'
                AND timestamp >= current_time + INTERVAL '5 minutes'
                ORDER BY timestamp ASC LIMIT 1
            ) next_5min
        ),
        -- Calculate fill probabilities
        fill_probs AS (
            SELECT
                COUNT(*) FILTER (WHERE outcome = 'FILLED' AND side = 'BID') / 
                    NULLIF(COUNT(*) FILTER (WHERE side = 'BID'), 0)::numeric AS fill_probability_bid,
                COUNT(*) FILTER (WHERE outcome = 'FILLED' AND side = 'ASK') / 
                    NULLIF(COUNT(*) FILTER (WHERE side = 'ASK'), 0)::numeric AS fill_probability_ask,
                AVG(time_to_outcome_seconds) FILTER (WHERE outcome = 'FILLED') AS avg_time_to_fill
            FROM order_outcomes
            WHERE creation_time BETWEEN current_time - INTERVAL '30 minutes' AND current_time
        )
        SELECT
            current_time,
            'WLD-USDT',
            cm.mid_price,
            cm.spread,
            cm.spread_pct,
            cm.bid_ask_imbalance,
            of.new_bid_orders,
            of.new_ask_orders,
            of.canceled_bid_orders,
            of.canceled_ask_orders,
            pc.price_change_1min,
            pc.price_change_5min,
            pc.price_volatility_1min,
            pc.next_mid_price_1min,
            pc.next_mid_price_5min,
            fp.fill_probability_bid,
            fp.fill_probability_ask,
            fp.avg_time_to_fill
        FROM current_market cm
        CROSS JOIN order_flow of
        CROSS JOIN price_changes pc
        CROSS JOIN fill_probs fp;
        
        -- Move to the next interval
        current_time := current_time + (interval_minutes || ' minutes')::interval;
    END LOOP;
END;
$$;

-- 10. Execute the procedure to populate TFT features for the available data range
CALL populate_tft_features(
    (SELECT MIN(timestamp) FROM order_events),
    (SELECT MAX(timestamp) FROM order_events),
    1 -- 1-minute intervals
);

-- 11. Create compression policy for TimescaleDB tables (optional but recommended)
SELECT add_compression_policy('order_events', INTERVAL '7 days');
SELECT add_compression_policy('order_book_summary', INTERVAL '7 days');
SELECT add_compression_policy('tft_features', INTERVAL '7 days'); 