-- Manual TFT features insertion
-- This adds a few sample rows to the existing tft_features table

-- Insert sample data rows
INSERT INTO tft_features 
(timestamp, trading_pair, mid_price, spread, spread_pct, imbalance, 
 new_bid_orders, new_ask_orders, canceled_bid_orders, canceled_ask_orders,
 next_price_1min, next_price_5min, fill_probability)
VALUES
('2025-04-24 00:00:00+00', 'WLD-USDT', 0.90500000, 0.00200000, 0.00220000, 0.05000000, 15, 12, 8, 7, 0.90600000, 0.90700000, 0.4500),
('2025-04-24 00:01:00+00', 'WLD-USDT', 0.90600000, 0.00210000, 0.00232000, 0.03000000, 12, 14, 6, 9, 0.90700000, 0.90800000, 0.4800),
('2025-04-24 00:02:00+00', 'WLD-USDT', 0.90700000, 0.00220000, 0.00242000, 0.02000000, 18, 10, 11, 5, 0.90800000, 0.90850000, 0.5200),
('2025-04-24 00:03:00+00', 'WLD-USDT', 0.90800000, 0.00210000, 0.00231000, 0.01000000, 14, 16, 7, 10, 0.90750000, 0.90800000, 0.5000),
('2025-04-24 00:04:00+00', 'WLD-USDT', 0.90750000, 0.00200000, 0.00220000, 0.04000000, 11, 9, 6, 4, 0.90700000, 0.90650000, 0.4700);

-- Query to verify data
SELECT * FROM tft_features ORDER BY timestamp; 