-- TFT Bot Database Schema
-- For use with TimescaleDB (PostgreSQL)

-- Order events table for tracking inferred order lifecycle
CREATE TABLE IF NOT EXISTS order_events (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    synthetic_order_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(20) NOT NULL,  -- 'NEW', 'MODIFY', 'CANCEL', 'FILL' (Inferred)
    event_timestamp TIMESTAMP NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    side VARCHAR(4) NOT NULL,  -- 'BID' or 'ASK'
    remaining_quantity DECIMAL(18, 8)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('order_events', 'event_timestamp', if_not_exists => TRUE);

-- Indices for efficient queries
CREATE INDEX IF NOT EXISTS order_events_order_id_idx
ON order_events (synthetic_order_id);

CREATE INDEX IF NOT EXISTS order_events_timestamp_pair_idx
ON order_events (event_timestamp, trading_pair);

-- Active orders tracking table
CREATE TABLE IF NOT EXISTS active_orders (
    synthetic_order_id VARCHAR(64) PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    side VARCHAR(4) NOT NULL,  -- 'BID' or 'ASK'
    status VARCHAR(20) NOT NULL  -- 'ACTIVE', 'CANCELED', 'FILLED' (Inferred)
);

CREATE INDEX IF NOT EXISTS active_orders_pair_price_idx
ON active_orders (trading_pair, price);

-- TFT model predictions table
CREATE TABLE IF NOT EXISTS tft_predictions (
    id SERIAL PRIMARY KEY,
    prediction_timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    dir_prediction DECIMAL(18, 8),
    down_prediction DECIMAL(18, 8),
    ensemble_prediction DECIMAL(18, 8),
    feature_context JSONB -- Store feature dict for analysis
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('tft_predictions', 'prediction_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS tft_predictions_timestamp_pair_idx
ON tft_predictions (prediction_timestamp, trading_pair);

-- Trading decisions table
CREATE TABLE IF NOT EXISTS trade_decisions (
    id SERIAL PRIMARY KEY,
    decision_timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    prediction_id INTEGER REFERENCES tft_predictions(id),
    decision VARCHAR(20) NOT NULL, -- 'ENTER_LONG', 'ENTER_SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD'
    target_price DECIMAL(18, 8), -- Price at decision time
    prediction_value DECIMAL(18, 8), -- Ensemble prediction value
    reason VARCHAR(100) -- e.g., 'Threshold crossed', 'Stop loss', 'Take profit', 'Max hold'
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('trade_decisions', 'decision_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS trade_decisions_timestamp_pair_idx
ON trade_decisions (decision_timestamp, trading_pair);

-- Executed trades table (when live trading enabled)
CREATE TABLE IF NOT EXISTS executed_trades (
    id SERIAL PRIMARY KEY,
    decision_id INTEGER REFERENCES trade_decisions(id),
    exchange_order_id VARCHAR(64) UNIQUE,
    trading_pair VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'BUY' or 'SELL'
    order_type VARCHAR(20) NOT NULL, -- 'MARKET', 'LIMIT'
    status VARCHAR(20) NOT NULL, -- 'FILLED', 'PARTIALLY_FILLED', 'CANCELED', 'REJECTED', 'NEW'
    requested_quantity DECIMAL(18, 8),
    filled_quantity DECIMAL(18, 8),
    average_fill_price DECIMAL(18, 8),
    commission DECIMAL(18, 8),
    commission_asset VARCHAR(10),
    transaction_time TIMESTAMP, -- Fill time from exchange
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('executed_trades', 'transaction_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS executed_trades_decision_id_idx
ON executed_trades (decision_id);

-- Position state tracking table
CREATE TABLE IF NOT EXISTS position_state (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL UNIQUE,
    position VARCHAR(5) NOT NULL, -- 'LONG', 'SHORT', 'NONE'
    entry_price DECIMAL(18, 8),
    position_size DECIMAL(18, 8),
    entry_timestamp TIMESTAMP,
    holding_periods INTEGER DEFAULT 0, -- Number of intervals held
    last_update_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics tracking table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    calculation_timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR(20) NOT NULL, -- '1h', '1d', '7d', 'all'
    total_trades INTEGER,
    win_rate DECIMAL(10, 6),
    profit_loss DECIMAL(18, 8),
    sharpe_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    avg_holding_periods DECIMAL(10, 6)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('performance_metrics', 'calculation_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS performance_metrics_timestamp_pair_idx
ON performance_metrics (calculation_timestamp, trading_pair); 