-- TFT Bot Database Schema (Improved for TimescaleDB)
-- For use with TimescaleDB (PostgreSQL)

-- Order events table for tracking inferred order lifecycle
DROP TABLE IF EXISTS order_events CASCADE;
CREATE TABLE IF NOT EXISTS order_events (
    id BIGSERIAL NOT NULL,
    trading_pair TEXT NOT NULL,
    synthetic_order_id TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'NEW', 'MODIFY', 'CANCEL', 'FILL' (Inferred)
    event_timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    side TEXT NOT NULL,  -- 'BID' or 'ASK'
    remaining_quantity DECIMAL(18, 8),
    PRIMARY KEY (id, event_timestamp)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('order_events', 'event_timestamp', if_not_exists => TRUE);

-- Indices for efficient queries
CREATE INDEX IF NOT EXISTS order_events_order_id_idx
ON order_events (synthetic_order_id);

CREATE INDEX IF NOT EXISTS order_events_timestamp_pair_idx
ON order_events (event_timestamp, trading_pair);

-- Active orders tracking table
DROP TABLE IF EXISTS active_orders CASCADE;
CREATE TABLE IF NOT EXISTS active_orders (
    synthetic_order_id TEXT PRIMARY KEY,
    trading_pair TEXT NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    side TEXT NOT NULL,  -- 'BID' or 'ASK'
    status TEXT NOT NULL  -- 'ACTIVE', 'CANCELED', 'FILLED' (Inferred)
);

CREATE INDEX IF NOT EXISTS active_orders_pair_price_idx
ON active_orders (trading_pair, price);

-- TFT model predictions table
DROP TABLE IF EXISTS tft_predictions CASCADE;
CREATE TABLE IF NOT EXISTS tft_predictions (
    id BIGSERIAL NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    trading_pair TEXT NOT NULL,
    dir_prediction DECIMAL(18, 8),
    down_prediction DECIMAL(18, 8),
    ensemble_prediction DECIMAL(18, 8),
    feature_context JSONB, -- Store feature dict for analysis
    PRIMARY KEY (id, prediction_timestamp)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('tft_predictions', 'prediction_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS tft_predictions_timestamp_pair_idx
ON tft_predictions (prediction_timestamp, trading_pair);

-- Trading decisions table
DROP TABLE IF EXISTS trade_decisions CASCADE;
CREATE TABLE IF NOT EXISTS trade_decisions (
    id BIGSERIAL NOT NULL,
    decision_timestamp TIMESTAMPTZ NOT NULL,
    trading_pair TEXT NOT NULL,
    prediction_id BIGINT,
    decision TEXT NOT NULL, -- 'ENTER_LONG', 'ENTER_SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD'
    target_price DECIMAL(18, 8), -- Price at decision time
    prediction_value DECIMAL(18, 8), -- Ensemble prediction value
    reason TEXT, -- e.g., 'Threshold crossed', 'Stop loss', 'Take profit', 'Max hold'
    PRIMARY KEY (id, decision_timestamp)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('trade_decisions', 'decision_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS trade_decisions_timestamp_pair_idx
ON trade_decisions (decision_timestamp, trading_pair);

-- Executed trades table (when live trading enabled)
DROP TABLE IF EXISTS executed_trades CASCADE;
CREATE TABLE IF NOT EXISTS executed_trades (
    id BIGSERIAL NOT NULL,
    decision_id BIGINT,
    exchange_order_id TEXT UNIQUE,
    trading_pair TEXT NOT NULL,
    side TEXT NOT NULL, -- 'BUY' or 'SELL'
    order_type TEXT NOT NULL, -- 'MARKET', 'LIMIT'
    status TEXT NOT NULL, -- 'FILLED', 'PARTIALLY_FILLED', 'CANCELED', 'REJECTED', 'NEW'
    requested_quantity DECIMAL(18, 8),
    filled_quantity DECIMAL(18, 8),
    average_fill_price DECIMAL(18, 8),
    commission DECIMAL(18, 8),
    commission_asset TEXT,
    transaction_time TIMESTAMPTZ NOT NULL, -- Fill time from exchange
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, transaction_time)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('executed_trades', 'transaction_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS executed_trades_decision_id_idx
ON executed_trades (decision_id);

-- Position state tracking table
DROP TABLE IF EXISTS position_state CASCADE;
CREATE TABLE IF NOT EXISTS position_state (
    id BIGSERIAL PRIMARY KEY,
    trading_pair TEXT NOT NULL UNIQUE,
    position TEXT NOT NULL, -- 'LONG', 'SHORT', 'NONE'
    entry_price DECIMAL(18, 8),
    position_size DECIMAL(18, 8),
    entry_timestamp TIMESTAMPTZ,
    holding_periods INTEGER DEFAULT 0, -- Number of intervals held
    last_update_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics tracking table
DROP TABLE IF EXISTS performance_metrics CASCADE;
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL NOT NULL,
    trading_pair TEXT NOT NULL,
    calculation_timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT NOT NULL, -- '1h', '1d', '7d', 'all'
    total_trades INTEGER,
    win_rate DECIMAL(10, 6),
    profit_loss DECIMAL(18, 8),
    sharpe_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    avg_holding_periods DECIMAL(10, 6),
    PRIMARY KEY (id, calculation_timestamp)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('performance_metrics', 'calculation_timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS performance_metrics_timestamp_pair_idx
ON performance_metrics (calculation_timestamp, trading_pair); 