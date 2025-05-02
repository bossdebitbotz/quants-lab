-- TimescaleDB TFT Feature Generation Examples
-- Shows how to use TimescaleDB functions for TFT-relevant time series operations

-- 1. Basic time bucketing (grouping by time intervals)
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    AVG(mid_price) AS avg_price,
    MAX(mid_price) - MIN(mid_price) AS price_range,
    COUNT(*) AS num_samples
FROM tft_features
GROUP BY bucket
ORDER BY bucket;

-- 2. Rolling window calculations
-- This calculates the 5-minute moving average price
WITH windowed_data AS (
    SELECT 
        timestamp,
        mid_price,
        AVG(mid_price) OVER (
            ORDER BY timestamp
            RANGE BETWEEN '5 minutes' PRECEDING AND CURRENT ROW
        ) AS price_moving_avg_5min
    FROM tft_features
)
SELECT * FROM windowed_data
ORDER BY timestamp;

-- 3. Calculate price volatility
SELECT 
    time_bucket('2 minutes', timestamp) AS bucket,
    trading_pair,
    STDDEV(mid_price) AS price_volatility,
    AVG(mid_price) AS avg_price
FROM tft_features
GROUP BY bucket, trading_pair
ORDER BY bucket;

-- 4. Calculate bid-ask imbalance statistics
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    AVG(imbalance) AS avg_imbalance,
    STDDEV(imbalance) AS imbalance_volatility,
    CORR(imbalance, mid_price) AS imbalance_price_correlation
FROM tft_features
GROUP BY bucket
ORDER BY bucket;

-- 5. Feature extraction for TFT - Combining time buckets and lookback windows
WITH base_features AS (
    SELECT 
        time_bucket('1 minute', timestamp) AS bucket,
        -- Price features
        AVG(mid_price) AS price,
        STDDEV(mid_price) AS price_volatility,
        AVG(spread) AS spread,
        -- Order flow features  
        SUM(new_bid_orders) AS bid_volume,
        SUM(new_ask_orders) AS ask_volume,
        SUM(canceled_bid_orders) AS canceled_bid_volume,
        SUM(canceled_ask_orders) AS canceled_ask_volume,
        -- Imbalance features
        AVG(imbalance) AS imbalance
    FROM tft_features
    GROUP BY bucket
)
SELECT 
    bucket,
    price,
    -- TFT needs historical context features - lookback values at different horizons
    LAG(price, 1) OVER (ORDER BY bucket) AS price_lag1,
    LAG(price, 2) OVER (ORDER BY bucket) AS price_lag2,
    LAG(price, 3) OVER (ORDER BY bucket) AS price_lag3,
    -- Spread and volatility features
    price_volatility,
    spread,
    -- Order flow and imbalance
    bid_volume - ask_volume AS order_flow_imbalance,
    (bid_volume - canceled_bid_volume) - (ask_volume - canceled_ask_volume) AS net_order_flow,
    imbalance,
    -- Target variable (using lead to look ahead)
    LEAD(price, 1) OVER (ORDER BY bucket) AS next_price_1min,
    LEAD(price, 5) OVER (ORDER BY bucket) AS next_price_5min
FROM base_features
ORDER BY bucket;

-- 6. Feature normalization - important for TFT model training
WITH feature_stats AS (
    SELECT 
        AVG(mid_price) AS avg_price,
        STDDEV(mid_price) AS std_price,
        AVG(spread) AS avg_spread,
        STDDEV(spread) AS std_spread,
        AVG(imbalance) AS avg_imbalance,
        STDDEV(imbalance) AS std_imbalance
    FROM tft_features
)
SELECT 
    timestamp,
    -- Z-score normalization for continuous features
    (mid_price - avg_price) / NULLIF(std_price, 0) AS price_normalized,
    (spread - avg_spread) / NULLIF(std_spread, 0) AS spread_normalized,
    (imbalance - avg_imbalance) / NULLIF(std_imbalance, 0) AS imbalance_normalized,
    -- Binary features often don't need normalization but can be scaled
    new_bid_orders::float / 20 AS new_bid_orders_scaled,
    new_ask_orders::float / 20 AS new_ask_orders_scaled
FROM tft_features, feature_stats
ORDER BY timestamp; 