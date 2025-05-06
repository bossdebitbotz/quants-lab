-- Fix the executed_trades table
DROP TABLE IF EXISTS executed_trades CASCADE;

CREATE TABLE IF NOT EXISTS executed_trades (
    id BIGSERIAL NOT NULL,
    decision_id BIGINT,
    exchange_order_id TEXT,
    trading_pair TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_quantity DECIMAL(18, 8),
    filled_quantity DECIMAL(18, 8),
    average_fill_price DECIMAL(18, 8),
    commission DECIMAL(18, 8),
    commission_asset TEXT,
    transaction_time TIMESTAMPTZ NOT NULL,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, transaction_time)
);

-- Create hypertable
SELECT create_hypertable('executed_trades', 'transaction_time', if_not_exists => TRUE);

-- Create indices
CREATE INDEX IF NOT EXISTS executed_trades_decision_id_idx ON executed_trades (decision_id);
CREATE INDEX IF NOT EXISTS executed_trades_order_id_idx ON executed_trades (exchange_order_id);
CREATE INDEX IF NOT EXISTS executed_trades_pair_time_idx ON executed_trades (trading_pair, transaction_time);

-- Add foreign key references
ALTER TABLE trade_decisions ADD CONSTRAINT trade_decisions_prediction_fk
    FOREIGN KEY (prediction_id) REFERENCES tft_predictions(id) ON DELETE SET NULL;

ALTER TABLE executed_trades ADD CONSTRAINT executed_trades_decision_fk
    FOREIGN KEY (decision_id) REFERENCES trade_decisions(id) ON DELETE SET NULL; 