# TFT Trading Bot Checklist

## Position Management
- [ ] Position state is correctly persisted in database
   - **Verification:** `python tools/check_position_state.py`
   - **Expected:** Position state should show status, entry price, position size, entry timestamp
   - **Files:** `utils/db_logger.py > update_position_state()`, `utils/db_logger.py > get_position_state()`

- [ ] Position updates are logged (LONG, SHORT, NONE)
   - **Verification:** `tail -100 logs/tft_bot.log | grep "position state"`
   - **Expected:** Should see log entries for position updates with state transitions
   - **Files:** `core/position_manager.py`

- [ ] Entry price, position size and timestamp are recorded
   - **Verification:** `python tools/check_position_state.py`
   - **Expected:** All fields should be populated when position is LONG or SHORT
   - **Database:** `SELECT * FROM position_state ORDER BY last_update_timestamp DESC LIMIT 1;`

- [ ] Holding periods are tracked and incremented
   - **Verification:** Monitor position_state table and check holding_periods field
   - **Expected:** holding_periods should increment periodically when position is open
   - **Files:** `utils/db_logger.py > increment_holding_period()`

- [ ] State is properly loaded between prediction cycles
   - **Verification:** `python tools/monitor_positions.py --interval 10`
   - **Expected:** State should persist between bot restarts and prediction cycles
   - **Files:** `core/realtime_tft_bot.py > load_position_state()`, `core/decision_engine.py`

- [ ] Position closing correctly resets position state
   - **Verification:** After position closes, check if state is NONE
   - **Command:** `python tools/check_position_state.py` after a trade closes
   - **Expected:** Position should be NONE with NULL entry price/size after closing

## Trade Execution
- [ ] Trade orders are sent to exchange correctly
   - **Verification:** Compare logs with exchange order history
   - **Command:** `tail -100 logs/execution.log | grep "sending order"`
   - **Files:** `execution/execution_handler.py > execute_order()`

- [ ] Order status is monitored until completion
   - **Verification:** Check logs for order status updates
   - **Command:** `tail -100 logs/execution.log | grep "order status"`
   - **Files:** `execution/execution_handler.py > monitor_order_status()`

- [ ] Failed orders are handled gracefully
   - **Verification:** Intentionally cause order failure (e.g., invalid size)
   - **Expected:** Bot should log failure, not crash, and handle error state
   - **Files:** `execution/execution_handler.py > handle_execution_error()`

- [ ] Position sizing is calculated according to config
   - **Verification:** Check logs for position size calculation
   - **Command:** `tail -100 logs/tft_bot.log | grep "position size"`
   - **Files:** `core/decision_engine.py > calculate_position_size()`

- [ ] Stop-loss and take-profit are set if configured
   - **Verification:** Check exchange for SL/TP orders after position open
   - **Files:** `execution/execution_handler.py > set_stop_loss_take_profit()`

- [ ] Trades are executed with the correct direction and size
   - **Verification:** Compare trade logs with exchange confirmation
   - **Command:** `python tools/pnl_tracker.py` to see recent trades
   - **Files:** `execution/execution_handler.py`

## PnL Tracking
- [ ] Entry and exit trades are properly paired
   - **Verification:** `python tools/pnl_summary.py --db your_database.db`
   - **Expected:** Complete position history with entry/exit pairs
   - **Files:** `utils/pnl_tracker.py > pair_trades()`

- [ ] PnL calculation is accurate with scientific notation handling
   - **Verification:** Manually calculate PnL and compare with reported
   - **Command:** `python tools/pnl_summary.py --db your_database.db`
   - **Files:** `utils/pnl_tracker.py > calculate_pnl_stats()`

- [ ] Trade history is complete and consistent
   - **Verification:** `python tools/pnl_summary.py --db your_database.db --verbose`
   - **Database:** `SELECT * FROM trade_decisions td LEFT JOIN executed_trades et ON td.id = et.decision_id ORDER BY td.timestamp DESC LIMIT 20;`
   - **Expected:** No missing trades or inconsistent data

- [ ] Performance metrics are updated after trades
   - **Verification:** Check performance_metrics table after closing trades
   - **Database:** `SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1;`
   - **Files:** `utils/db_logger.py > log_trade()`

- [ ] Cumulative PnL is correctly maintained
   - **Verification:** Compare sum of individual PnLs with cumulative
   - **Command:** `python tools/pnl_summary.py --db your_database.db`
   - **Expected:** Cumulative PnL matches sum of individual trade PnLs
   - **Note:** Use the standardized PnL tracking utility for consistent results

## Decision Engine
- [ ] Model predictions are loaded correctly
   - **Verification:** Check logs for prediction values
   - **Command:** `tail -100 logs/tft_bot.log | grep "prediction"`
   - **Files:** `core/tft_predictor.py > predict()`

- [ ] Ensemble predictions combine models appropriately
   - **Verification:** Check logs for individual and ensemble predictions
   - **Command:** `tail -100 logs/tft_bot.log | grep "ensemble"`
   - **Files:** `core/tft_predictor.py > ensemble_predictions()`

- [ ] Trading thresholds are enforced
   - **Verification:** Compare prediction values vs trade decisions
   - **Database:** `SELECT prediction_value, decision FROM trade_decisions ORDER BY timestamp DESC LIMIT 10;`
   - **Files:** `core/decision_engine.py > should_trade()`

- [ ] Current position state affects trading decisions
   - **Verification:** Check decision logs with different position states
   - **Expected:** Different decisions for same prediction depending on current position
   - **Files:** `core/decision_engine.py > make_decision()`

- [ ] Max holding period limits are enforced
   - **Verification:** Monitor position that reaches max holding period
   - **Expected:** Position should close after MAX_HOLDING_PERIOD cycles
   - **Files:** `core/decision_engine.py > check_holding_period_exit()`

- [ ] Stop-loss and take-profit rules are followed
   - **Verification:** Monitor position near SL/TP levels
   - **Expected:** Position should close when price hits SL/TP levels
   - **Files:** `core/decision_engine.py > check_sl_tp_exit()`

## TFT Model Verification
- [ ] Model files exist in specified paths
   - **Verification:** `ls -l ../models/*tft*`
   - **Expected:** Model files should exist with recent timestamps
   - **Config:** Check `config.json` for model path settings

- [ ] Model architecture matches expected configuration
   - **Verification:** Load model and print architecture summary
   - **Files:** `models/model_loader.py > load_model()`
   - **Expected:** Model architecture matches documentation

- [ ] Feature dimensions match model input requirements 
   - **Verification:** Compare feature generation with model input shape
   - **Files:** `feature_engineering/feature_generator.py`, `core/tft_predictor.py`
   - **Expected:** Feature vector dimensions match model input requirements

- [ ] Feature transformations are applied consistently
   - **Verification:** Log feature values before prediction
   - **Files:** `feature_engineering/transformations.py`
   - **Expected:** Features should be properly normalized/scaled

- [ ] Model returns predictions in expected format
   - **Verification:** Check prediction output structure
   - **Files:** `core/tft_predictor.py > predict()`
   - **Expected:** Predictions should be between -1 and 1

- [ ] Both directional and downward models operate correctly
   - **Verification:** Check logs for both model predictions
   - **Command:** `tail -100 logs/tft_bot.log | grep "dir model" | grep "down model"`
   - **Files:** `core/tft_predictor.py`

- [ ] Prediction values are within expected ranges
   - **Verification:** Check prediction logs for reasonable values
   - **Database:** `SELECT dir_prediction, down_prediction, ensemble_prediction FROM tft_predictions ORDER BY prediction_timestamp DESC LIMIT 20;`
   - **Expected:** Values typically between -0.05 and 0.05

- [ ] Models load without errors on startup
   - **Verification:** Check startup logs for model loading
   - **Command:** `grep "model loaded" logs/tft_bot.log`
   - **Files:** `core/tft_predictor.py > load_models()`

- [ ] Inference speed is adequate for prediction interval
   - **Verification:** Check logs for prediction duration
   - **Command:** `grep "prediction took" logs/tft_bot.log`
   - **Expected:** Prediction time significantly less than prediction interval

- [ ] Historical vs. real-time prediction consistency
   - **Verification:** Compare backtest predictions vs. real-time on same data
   - **Files:** `backtesting/backtest_validator.py`
   - **Expected:** Similar prediction values for same input data

## Database Operations
- [ ] Connection pool is managed efficiently
   - **Verification:** Check db connection logs
   - **Files:** `utils/db_logger.py > _get_connection()`, `utils/db_logger.py > _return_connection()`
   - **Expected:** No connection leaks or pool exhaustion

- [ ] Predictions are logged to database
   - **Verification:** `SELECT * FROM tft_predictions ORDER BY prediction_timestamp DESC LIMIT 10;`
   - **Files:** `utils/db_logger.py > log_prediction()`
   - **Expected:** Regular prediction entries with complete data

- [ ] Trade decisions are recorded with reasons
   - **Verification:** `SELECT * FROM trade_decisions ORDER BY timestamp DESC LIMIT 10;`
   - **Files:** `utils/db_logger.py > log_trade_decision()`
   - **Expected:** Decision entries with explanatory reason field

- [ ] Position state is persisted between restarts
   - **Verification:** Restart bot and check if position state remains
   - **Command:** `python tools/check_position_state.py` before and after restart
   - **Files:** `core/realtime_tft_bot.py > load_position_state()`

- [ ] Database schema is up to date
   - **Verification:** `python tools/check_db_schema.py`
   - **Files:** `utils/db_logger.py > _create_tables()`, `utils/db_logger.py > _verify_position_state_schema()`
   - **Expected:** All required tables and columns present

- [ ] Transactions are properly committed
   - **Verification:** Check for transaction rollback errors in logs
   - **Files:** `utils/db_logger.py` (look for `conn.commit()` and `conn.rollback()`)
   - **Expected:** Transactions properly committed, no dangling transactions

## Configuration
- [ ] Configuration is loaded successfully
   - **Verification:** Check startup logs for config loading
   - **Command:** `grep "Configuration loaded" logs/tft_bot.log`
   - **Files:** `config.py > load_config()`

- [ ] API keys are securely handled
   - **Verification:** Check if keys are masked in logs
   - **Files:** `config.py`, ensure keys are from env vars or secure storage
   - **Expected:** No plaintext API keys in code or logs

- [ ] Model paths are valid
   - **Verification:** Check if model files exist at specified paths
   - **Command:** `grep "model path" logs/tft_bot.log`
   - **Config:** Check `config.json` for model path settings

- [ ] Trade parameters (thresholds, sizing) are reasonable
   - **Verification:** Review config settings against documentation
   - **Files:** `config.json`, `docs/configuration.md`
   - **Expected:** Parameters within recommended ranges

- [ ] Database connection settings are correct
   - **Verification:** Check database connection success logs
   - **Command:** `grep "Connected to database" logs/tft_bot.log`
   - **Config:** Check `config.json` for database settings

## Monitoring & Diagnostics
- [ ] Logs are detailed and informative
   - **Verification:** Review logs for key events
   - **Command:** `tail -100 logs/tft_bot.log`
   - **Expected:** Structured logs with appropriate detail level

- [ ] Position monitor shows current state
   - **Verification:** `python tools/monitor_positions.py`
   - **Expected:** Real-time updates of position state changes
   - **Files:** `tools/monitor_positions.py`

- [ ] PnL tracker shows accurate trade history
   - **Verification:** `python tools/pnl_summary.py --db your_database.db`
   - **Expected:** Complete trade history with accurate PnL
   - **Files:** `utils/pnl_tracker.py`, `tools/pnl_summary.py`
   - **Note:** Bot monitor and PnL summary now use the same standardized tracking logic

- [ ] Errors are caught and logged appropriately
   - **Verification:** Check error handling in logs
   - **Command:** `grep "ERROR" logs/tft_bot.log`
   - **Expected:** Detailed error logs with context

## Error Handling
- [ ] API connection errors are handled
   - **Verification:** Simulate API error and check response
   - **Files:** `execution/api_client.py` error handling code
   - **Expected:** Retry logic and graceful degradation

- [ ] Database connection failures are managed
   - **Verification:** Temporarily disable DB and check response
   - **Files:** `utils/db_logger.py` error handling code
   - **Expected:** Retry logic and appropriate error logs

- [ ] Model prediction errors don't crash the bot
   - **Verification:** Corrupt model file or input data and check response
   - **Files:** `core/tft_predictor.py` error handling code
   - **Expected:** Fallback logic and clear error logs

- [ ] Trade execution failures are detected and reported
   - **Verification:** Cause order failure and check response
   - **Files:** `execution/execution_handler.py` error handling
   - **Expected:** Detailed error logs and notification

- [ ] Bot can recover from most error states
   - **Verification:** Introduce various failures and check recovery
   - **Expected:** Self-healing behavior for non-critical errors
   - **Files:** `core/realtime_tft_bot.py` error recovery code

## General System
- [ ] Bot starts successfully
   - **Verification:** `python core/realtime_tft_bot.py`
   - **Expected:** Bot starts without errors
   - **Files:** `core/realtime_tft_bot.py`

- [ ] Prediction cycle runs at configured interval
   - **Verification:** Check timestamps between predictions
   - **Command:** `grep "prediction cycle" logs/tft_bot.log`
   - **Expected:** Regular intervals matching PREDICTION_INTERVAL_SECONDS

- [ ] No unexpected duplicate positions
   - **Verification:** `python tools/check_position_state.py`
   - **Database:** `SELECT * FROM trades WHERE action='OPEN' ORDER BY timestamp DESC LIMIT 10;`
   - **Expected:** No duplicate OPEN trades without corresponding CLOSE

- [ ] No missed position exits
   - **Verification:** Review position history for stuck positions
   - **Command:** `python tools/pnl_tracker.py`
   - **Expected:** All positions eventually closed

- [ ] Memory usage remains stable over time
   - **Verification:** Monitor process memory with `ps` or monitoring tool
   - **Expected:** No continuous memory growth
   - **Files:** Check for memory leaks in long-running processes

- [ ] CPU usage is reasonable
   - **Verification:** Monitor CPU usage during operation
   - **Command:** `top -pid $(pgrep -f realtime_tft_bot.py)`
   - **Expected:** CPU usage spikes only during prediction, not continuous

## Pre-Trade Verification
- [ ] Account has sufficient balance
   - **Verification:** Check account balance logs before trades
   - **Command:** `grep "account balance" logs/execution.log`
   - **Files:** `execution/execution_handler.py > check_balance()`

- [ ] Exchange connection is stable
   - **Verification:** Check exchange API ping times
   - **Command:** `grep "exchange latency" logs/execution.log`
   - **Files:** `execution/api_client.py > ping()`

- [ ] Position state matches exchange records
   - **Verification:** Compare local position state with exchange position
   - **Command:** `python tools/verify_exchange_positions.py`
   - **Expected:** Local state should match exchange position

- [ ] No orphaned/stuck orders exist
   - **Verification:** Check for old pending orders
   - **Command:** `python tools/check_stuck_orders.py`
   - **Files:** `execution/execution_handler.py > clean_stuck_orders()`

- [ ] Models are loaded and functioning
   - **Verification:** Check model loading logs
   - **Command:** `grep "model loaded" logs/tft_bot.log`
   - **Files:** `core/tft_predictor.py > load_models()`

## Post-Trade Verification
- [ ] Trade was executed at expected price and size
   - **Verification:** Compare order request vs execution confirmation
   - **Database:** `SELECT * FROM executed_trades ORDER BY transaction_time DESC LIMIT 1;`
   - **Files:** `execution/execution_handler.py > confirm_execution()`

- [ ] Position state updated correctly
   - **Verification:** `python tools/check_position_state.py` after trade
   - **Expected:** Position state should reflect recent trade
   - **Files:** `core/position_manager.py > update_position()`

- [ ] PnL recorded accurately
   - **Verification:** `python tools/pnl_tracker.py` after position close
   - **Expected:** PnL calculation matches (exit_price - entry_price) * size
   - **Files:** `utils/db_logger.py > log_trade()`

- [ ] Logs show appropriate details about trade
   - **Verification:** Check logs around trade execution time
   - **Command:** `grep "trade executed" logs/tft_bot.log`
   - **Expected:** Complete details about the trade

- [ ] Database state is consistent with bot memory
   - **Verification:** Compare in-memory state with database state
   - **Command:** `python tools/verify_state_consistency.py`
   - **Expected:** Database state matches bot's internal state

## Scheduled Maintenance Tasks
- [ ] Check for database size/growth
   - **Verification:** Monitor database size 
   - **Command:** `du -h /path/to/database` or check database metrics
   - **Expected:** Linear growth, no unexpected size increases

- [ ] Verify log rotation
   - **Verification:** Check log files for rotation
   - **Command:** `ls -l logs/`
   - **Expected:** Old logs archived, no oversized log files

- [ ] Check model performance metrics
   - **Verification:** Review prediction accuracy over time
   - **Command:** `python tools/analyze_model_performance.py`
   - **Expected:** Consistent prediction quality

- [ ] Review recent trade performance
   - **Verification:** `python tools/pnl_tracker.py --days 30`
   - **Expected:** Performance metrics within expected ranges
   - **Files:** `tools/pnl_tracker.py`

- [ ] Verify bot uptime and stability
   - **Verification:** Check process uptime
   - **Command:** `ps -p $(pgrep -f realtime_tft_bot.py) -o etime=`
   - **Expected:** Continuous operation without unexpected restarts

### Data Analysis and Monitoring

- [x] Run `tools/bot_monitor.py` to view current state
- [x] Check executed trades with `tools/pnl_summary.py --db <path_to_database.db>` (preferred tool for PnL analysis)
      Note: This tool properly handles scientific notation price formats (0E-8)
- [x] Verify position state with `tools/check_position_state.py`
- [x] Examine database schema with `tools/check_db_schema.py`
- [x] Check database tables with `tools/check_trade_decisions.py` and `tools/check_executed_trades.py` 