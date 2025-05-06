# TFT Bot Backtesting Framework

This framework allows you to backtest the TFT bot against historical order book data collected by the `order_lifecycle_tracker.py` script.

## Overview

The backtesting framework uses the same feature calculation and prediction pipeline as the live TFT bot but applies it to historical data. This ensures that backtest results are as close as possible to what you would expect in live trading.

The framework:
1. Retrieves historical order book data from the PostgreSQL database
2. Reconstructs OHLCV bars from the order book data
3. Calculates the same features used by the live bot
4. Generates predictions using the TFT models
5. Simulates trading based on these predictions
6. Calculates performance metrics and generates visualizations

## Features

- **Exact Bot Replication**: Uses the exact same feature calculation and prediction code as the live bot
- **Order Book Data**: Works with real order book data captured by `order_lifecycle_tracker.py`
- **Trading Simulation**: Includes realistic trading simulation with position sizing, take-profit, and stop-loss
- **Performance Metrics**: Calculates comprehensive performance metrics (return, Sharpe ratio, drawdown, etc.)
- **Visualization**: Generates charts for price, portfolio value, trading signals, and model predictions

## Requirements

- PostgreSQL database with order book data collected by `order_lifecycle_tracker.py`
- All dependencies required by the TFT bot
- Python 3.8 or higher

## Usage

### Running a Backtest

To run a backtest with default settings:

```bash
python backtesting/run_backtest.py
```

This will backtest the most recent 7 days of available data using default trading parameters.

### Command Line Options

You can customize the backtest using various command line options:

```bash
python backtesting/run_backtest.py --trading-pair WLD-USDT --days 3 --initial-balance 10000 --position-size 0.2
```

Available options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--trading-pair` | Trading pair to backtest | WLD-USDT |
| `--start-date` | Start date (YYYY-MM-DD) | Last 7 days |
| `--end-date` | End date (YYYY-MM-DD) | Today |
| `--days` | Number of days to backtest | 7 |
| `--initial-balance` | Initial balance in quote currency | 10000 |
| `--trade-threshold` | Threshold for trading signals | 0.0005 |
| `--position-size` | Position size as percentage of balance | 0.1 (10%) |
| `--take-profit` | Take profit percentage | 0.01 (1%) |
| `--stop-loss` | Stop loss percentage | 0.005 (0.5%) |

### Output

The backtest generates the following output files:

- `backtesting/backtest_results.json`: Performance metrics in JSON format
- `backtesting/equity_curve.csv`: Detailed equity curve data
- `backtesting/trades.csv`: List of all trades executed
- `backtesting/predictions.csv`: All predictions made by the model
- `backtesting/backtest_results.png`: Visual chart of price, portfolio value, and signals
- `backtesting/predictions_vs_actual.png`: Comparison of predictions vs actual price

### Interpreting Results

The backtest outputs the following key performance metrics:

- **Total Return**: Overall percentage return
- **Annualized Return**: Return normalized to an annual basis
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest percentage drop from peak to trough
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of trades executed

## Customization

You can further customize the backtest by modifying the `tft_orderbook_backtest.py` file:

- Change the trading logic in the `run_backtest` method
- Modify how OHLCV data is constructed from order book events
- Implement different position sizing or risk management strategies
- Add additional performance metrics

## Database Integration

The backtest framework connects to the same PostgreSQL database that was used by the `order_lifecycle_tracker.py` script. Make sure the database is running and accessible with the configured credentials.

Default database configuration:
- Host: localhost
- Port: 5438
- User: backtest_user
- Password: backtest_password
- Database: backtest_db

## Limitations

- The backtest assumes perfect execution at the close price of each bar
- The current implementation only supports long positions
- Take-profit and stop-loss are checked at bar close, not intra-bar

## Future Improvements

- Add support for short positions
- Implement intra-bar take-profit and stop-loss checks
- Add slippage simulation based on order book depth
- Support for custom trading strategies
- Parallel backtesting of multiple parameter combinations 