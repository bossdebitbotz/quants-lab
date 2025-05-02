-- Super minimal TimescaleDB test script

-- Create a simple test table
CREATE TABLE IF NOT EXISTS test_timescale (
    time TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION,
    sensor_id INTEGER
);

-- Convert to hypertable
SELECT create_hypertable('test_timescale', 'time', if_not_exists => TRUE);

-- Insert some test data
INSERT INTO test_timescale (time, value, sensor_id)
SELECT 
    NOW() - (INTERVAL '1 minute' * g),
    random() * 100,
    (random() * 5)::int
FROM generate_series(1, 100) g;

-- Query to verify data
SELECT 
    time_bucket('1 minute', time) AS minute,
    AVG(value) AS avg_value,
    COUNT(*) 
FROM test_timescale
GROUP BY minute
ORDER BY minute DESC
LIMIT 10; 