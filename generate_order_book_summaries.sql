-- Generate Order Book Summaries from Raw Order Events
-- This script fills in missing order book summary data using order events table
-- to allow for training with the full 7 days of data

-- 1. First, identify the time range we need to fill
WITH time_bounds AS (
    SELECT 
        MAX(timestamp) AS last_summary_time,
        (SELECT MAX(timestamp) FROM order_events) AS last_event_time
    FROM order_book_summary
)
SELECT 
    last_summary_time,
    last_event_time,
    (EXTRACT(EPOCH FROM (last_event_time - last_summary_time)) / 3600)::numeric(10,2) AS hours_to_fill
FROM time_bounds;

-- 2. Generate 1-minute intervals for the entire period we need to fill
WITH time_intervals AS (
    SELECT 
        generate_series(
            (SELECT date_trunc('minute', MAX(timestamp)) FROM order_book_summary), 
            (SELECT date_trunc('minute', MAX(timestamp)) FROM order_events),
            '1 minute'::interval
        ) AS timestamp
)
SELECT COUNT(*) FROM time_intervals;

-- 3. Create a temporary table to hold the synthesized order book summaries
DROP TABLE IF EXISTS temp_order_book_summary;
CREATE TEMP TABLE temp_order_book_summary AS 
WITH 
-- Generate time points at 1-minute intervals over the period we need to fill
time_points AS (
    SELECT 
        generate_series(
            (SELECT date_trunc('minute', MAX(timestamp)) + INTERVAL '1 minute' FROM order_book_summary), 
            (SELECT date_trunc('minute', MAX(timestamp)) FROM order_events),
            '1 minute'::interval
        ) AS timestamp
),
-- For each time point, identify the active bids and asks
active_bids_at_time AS (
    SELECT 
        tp.timestamp,
        ao.price,
        SUM(ao.quantity) AS total_quantity
    FROM time_points tp
    JOIN active_orders ao ON 
        ao.side = 'BID' AND 
        ao.status = 'ACTIVE' AND
        ao.last_updated <= tp.timestamp
    GROUP BY tp.timestamp, ao.price
),
active_asks_at_time AS (
    SELECT 
        tp.timestamp,
        ao.price,
        SUM(ao.quantity) AS total_quantity
    FROM time_points tp
    JOIN active_orders ao ON 
        ao.side = 'ASK' AND 
        ao.status = 'ACTIVE' AND
        ao.last_updated <= tp.timestamp
    GROUP BY tp.timestamp, ao.price
),
-- Calculate order book metrics for each time point
order_book_metrics AS (
    SELECT 
        tp.timestamp,
        -- Best bid - highest price among active bids
        (SELECT MAX(price) FROM active_bids_at_time WHERE timestamp = tp.timestamp) AS best_bid,
        -- Best ask - lowest price among active asks
        (SELECT MIN(price) FROM active_asks_at_time WHERE timestamp = tp.timestamp) AS best_ask,
        -- Total bid quantity
        (SELECT SUM(total_quantity) FROM active_bids_at_time WHERE timestamp = tp.timestamp) AS bid_quantity,
        -- Total ask quantity
        (SELECT SUM(total_quantity) FROM active_asks_at_time WHERE timestamp = tp.timestamp) AS ask_quantity
    FROM time_points tp
)
SELECT 
    obm.timestamp,
    'WLD-USDT' AS trading_pair,
    obm.best_bid,
    obm.best_ask,
    -- Calculate mid price
    (obm.best_bid + obm.best_ask) / 2 AS mid_price,
    -- Calculate spread
    (obm.best_ask - obm.best_bid) AS spread,
    -- Calculate spread percentage
    ((obm.best_ask - obm.best_bid) / ((obm.best_bid + obm.best_ask) / 2)) AS spread_pct,
    -- Calculate bid quantity
    COALESCE(obm.bid_quantity, 0) AS bid_quantity,
    -- Calculate ask quantity
    COALESCE(obm.ask_quantity, 0) AS ask_quantity,
    -- Calculate imbalance
    CASE 
        WHEN COALESCE(obm.bid_quantity, 0) + COALESCE(obm.ask_quantity, 0) > 0 THEN
            (COALESCE(obm.bid_quantity, 0) - COALESCE(obm.ask_quantity, 0)) / 
            (COALESCE(obm.bid_quantity, 0) + COALESCE(obm.ask_quantity, 0))
        ELSE 0
    END AS imbalance
FROM order_book_metrics obm
WHERE obm.best_bid IS NOT NULL AND obm.best_ask IS NOT NULL;

-- Check how many records we generated
SELECT COUNT(*) FROM temp_order_book_summary;
SELECT MIN(timestamp), MAX(timestamp) FROM temp_order_book_summary;

-- 4. Insert the synthesized data into the order_book_summary table
-- First filter to only include timestamps that don't already exist in the table
INSERT INTO order_book_summary (
    timestamp, trading_pair, best_bid, best_ask, mid_price, 
    spread, spread_pct, bid_quantity, ask_quantity, imbalance
)
SELECT 
    t.timestamp, t.trading_pair, t.best_bid, t.best_ask, t.mid_price, 
    t.spread, t.spread_pct, t.bid_quantity, t.ask_quantity, t.imbalance
FROM temp_order_book_summary t
WHERE NOT EXISTS (
    SELECT 1 FROM order_book_summary obs 
    WHERE obs.timestamp = t.timestamp AND obs.trading_pair = t.trading_pair
);

-- 5. Check the updated order_book_summary table
SELECT COUNT(*) FROM order_book_summary;
SELECT MIN(timestamp), MAX(timestamp) FROM order_book_summary;

-- 6. Now run the TFT feature preparation to use the extended data
-- This requires running the tft_prepare.sql script afterward 