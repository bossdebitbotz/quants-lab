# TFT Bot Monitoring Tool

This tool provides a command-line UI for monitoring the TFT trading bot. It displays real-time information about positions, trades, model predictions, and system status in a consolidated view.

## Features

- **Real-time Position Monitoring**: View current position state, entry price, position size, and holding periods
- **Recent Trades**: See the most recent trades executed by the bot with PnL information
- **Model Predictions**: View recent model predictions from directional and downward models
- **PnL Statistics**: Track performance metrics including total PnL, win rate, and trade count
- **System Status**: Monitor bot running status and view recent error messages
- **Automatic Refreshing**: Data automatically refreshes at configurable intervals

## Requirements

- Python 3.6+
- Required Python packages:
  - pandas
  - curses (usually included with Python)
  - psutil (for process monitoring)
  - tabulate
  - psycopg2-binary (PostgreSQL database adapter)
  - python-dotenv (for loading environment variables)
  - pyyaml (for config files)
  - numpy
  - torch (for TFT model processing)

## Installation

Run the setup script to install all required dependencies:

```bash
./setup_monitor.sh
```

If you prefer to install dependencies manually:

```bash
python3 -m pip install pandas tabulate psutil psycopg2-binary python-dotenv pyyaml numpy torch
```

## Usage

After installing dependencies, run the monitor with:

```bash
./run_monitor.sh
```

Or with custom refresh interval (in seconds):

```bash
./run_monitor.sh --refresh 10
```

You can also run the Python script directly:

```bash
python3 bot_monitor.py --refresh 5
```

## Interface Controls

- **q**: Quit the monitor
- **r**: Manually refresh data

## Sections Explained

1. **Current Position**: Shows the active position (LONG/SHORT/NONE), entry price, size, entry time, and holding periods.

2. **Recent Trades**: Displays the last 5 trades with timestamp, action (OPEN/CLOSE), direction (BUY/SELL), price, size, and PnL (for CLOSE trades).

3. **Recent Predictions**: Shows the last 5 model predictions with timestamp and values from directional model, downward model, and ensemble.

4. **PnL Statistics**: Provides summary statistics for trades over the last 7 days, including total PnL, number of trades, winning trades, and win rate.

5. **Recent Errors**: Displays the most recent error messages, if any.

## Troubleshooting

- If the monitor fails to start with dependency errors (like "No module named X"), run the setup script.
- If the monitor fails to start, check that the bot's database is accessible and the config.py file is properly set up.
- If data is not refreshing, verify that the database connection is working.
- Check the bot_monitor.log file for detailed error messages.

## Integration with Checklist

This monitor helps verify several items from the checklist.md:

- Position state tracking
- PnL calculation
- Trade execution
- Model prediction values
- System stability

For a complete monitoring checklist, refer to realtime_tft_bot/checklist.md. 