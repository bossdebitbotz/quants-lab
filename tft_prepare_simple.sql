-- Ultra-simplified TFT Features Script
-- Creates a basic table and inserts a few manually-specified rows

-- First, drop the table if it exists
DROP TABLE IF EXISTS tft_features;

-- Create a simplified TFT features table
CREATE TABLE tft_features (
    timestamp TIMESTAMPTZ NOT NULL,
    trading_pair TEXT NOT NULL,
    mid_price NUMERIC(18,8),
    spread NUMERIC(18,8),
    new_orders INTEGER,
    canceled_orders INTEGER,
    next_price_1min NUMERIC(18,8),
    PRIMARY KEY (timestamp, trading_pair)
);

-- Convert to hypertable
SELECT create_hypertable('tft_features', 'timestamp');

-- Insert some simple manual test data 
INSERT INTO tft_features VALUES
('2025-04-24 00:00:00', 'WLD-USDT', 0.90500000, 0.00200000, 10, 5, 0.90600000),
('2025-04-24 00:01:00', 'WLD-USDT', 0.90600000, 0.00210000, 8, 4, 0.90700000),
('2025-04-24 00:02:00', 'WLD-USDT', 0.90700000, 0.00220000, 12, 6, 0.90800000),
('2025-04-24 00:03:00', 'WLD-USDT', 0.90800000, 0.00210000, 7, 3, 0.90750000),
('2025-04-24 00:04:00', 'WLD-USDT', 0.90750000, 0.00200000, 9, 4, 0.90700000);

-- Verify data
SELECT * FROM tft_features ORDER BY timestamp; 