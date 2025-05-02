# TimescaleDB Setup for Temporal Fusion Transformer

This document outlines the setup and data preparation for using TimescaleDB with Temporal Fusion Transformer (TFT) models in our quantitative trading pipeline.

## Database Structure Setup

We've successfully set up TimescaleDB for time-series optimized storage of order book and trading data:

1. Confirmed TimescaleDB extension is properly installed (v2.19.3)
2. Created a time-series optimized table `tft_features` with timestamp partitioning
3. Inserted sample data to test TimescaleDB functionality
4. Created sample queries demonstrating time-bucketing and feature engineering

## Feature Transformations for TFT

The TFT model requires specific data organization:

1. **Regular time intervals** - Implemented via `time_bucket` function
2. **Historical context features** - Created using window functions and LAG operations
3. **Normalization** - Demonstrated Z-score normalization for continuous features
4. **Target variables** - Added future price columns using LEAD function

## Files Created

- `timescale_upgrade.sql` - Initial TimescaleDB upgrade script
- `fix_timescale_errors.sql` - Script to address TimescaleDB integration issues
- `minimal_setup.sql` - Minimal test case for TimescaleDB functionality
- `tft_prepare.sql` - Script to prepare TFT features table
- `tft_prepare_simple.sql` - Simplified version of TFT preparation
- `manual_tft_sample.sql` - Manual sample data insertion script
- `tft_sample_queries.sql` - Example queries for TFT feature engineering

## Current Status

✅ Successfully created TimescaleDB hypertable for TFT features  
✅ Demonstrated effective time-series querying and feature engineering  
✅ Added sample data for testing

## Next Steps

1. **Full Data Migration**
   - Run the complete feature generation pipeline on all historical data
   - Set up continuous data ingestion for real-time updates

2. **Feature Engineering Automation**
   - Create a scheduled job to update the TFT features at regular intervals
   - Implement materialized views for common feature combinations

3. **Model Training Integration**
   - Export feature data to training pipeline
   - Use continuous aggregates for efficient feature calculation

4. **Production Deployment**
   - Set up monitoring for data quality and completeness
   - Implement compression policies for long-term storage

## Sample TFT Query Pattern

```sql
-- Basic pattern for TFT feature extraction
WITH base_features AS (
    SELECT 
        time_bucket('1 minute', timestamp) AS bucket,
        -- Static features
        trading_pair,
        -- Observation features  
        AVG(mid_price) AS price,
        -- Context features
        SUM(new_bid_orders) AS bid_volume
    FROM tft_features
    GROUP BY bucket, trading_pair
),
tft_ready AS (
    SELECT 
        bucket,
        trading_pair,
        price,
        -- Historical context (lookback windows)
        LAG(price, 1) OVER w AS price_lag1,
        LAG(price, 5) OVER w AS price_lag5,
        -- Target variables (forecast horizons)
        LEAD(price, 1) OVER w AS target_1min,
        LEAD(price, 5) OVER w AS target_5min
    FROM base_features
    WINDOW w AS (PARTITION BY trading_pair ORDER BY bucket)
)
SELECT * FROM tft_ready
ORDER BY bucket;
``` 