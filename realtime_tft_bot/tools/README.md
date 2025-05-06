# Diagnostic Tools for TFT Bot

This directory contains diagnostic tools to help troubleshoot and monitor the TFT trading bot, particularly focused on position state tracking and performance analysis.

## Tools Overview

### 1. `check_position_state.py`

This tool checks the current position state in the database and can also manually set a position state.

**Usage:**

```bash
# Check current position state
python check_position_state.py

# Check position for a specific trading pair
python check_position_state.py --trading-pair BTC-USDT

# Set position state (POSITION,PRICE,SIZE format)
python check_position_state.py --set-position LONG,45000,0.1
python check_position_state.py --set-position NONE,none,none
```

### 2. `monitor_positions.py`

This tool continuously monitors position state changes and logs them to a file. It helps verify position state persistence and shows when positions are updated.

**Usage:**

```bash
# Start monitoring positions
python monitor_positions.py

# Monitor a specific trading pair with a custom interval
python monitor_positions.py --trading-pair ETH-USDT --interval 10
```

### 3. `pnl_tracker.py`

This tool analyzes the profitability of trades made by the TFT bot. It provides detailed statistics on PnL by position and overall trading performance.

**Usage:**

```bash
# View current position and recent trade history with PnL analysis
python pnl_tracker.py

# Analyze specific trading pair and time period
python pnl_tracker.py --pair BTC-USDT --days 30

# Export position data to CSV
python pnl_tracker.py --csv position_history.csv
```

**Output:**
- Current open position details
- Table of recently closed positions with PnL
- Performance statistics (win rate, total PnL, max drawdown, etc.)
- Option to export data to CSV for further analysis

## Troubleshooting Position Issues

If the bot is opening positions unexpectedly or not properly closing them:

1. Check the current position state using `check_position_state.py`
2. Monitor position changes with `monitor_positions.py` to see if position state is being maintained
3. Review PnL data with `pnl_tracker.py` to understand trade outcomes
4. If needed, manually set the position state using `check_position_state.py --set-position`

All tools read from the same database that the bot uses, so they accurately reflect the bot's internal state.

## Advanced Diagnostics

For more advanced diagnostics, consider running these SQL queries directly against the database:

```sql
-- Check position state
SELECT * FROM position_state;

-- Check recent decisions
SELECT * FROM trade_decisions ORDER BY timestamp DESC LIMIT 10;

-- Check executed trades
SELECT * FROM executed_trades ORDER BY timestamp DESC LIMIT 10;
```

## Documentation

For more information on the bot architecture and components, refer to:
- `docs/architecture.md` - System architecture and component interactions
- `docs/technical.md` - Technical implementation details
- `docs/troubleshooting.md` - Common issues and solutions 