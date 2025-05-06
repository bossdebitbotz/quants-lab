# Real-Time TFT Bot for Crypto Trading

A real-time trading bot that uses ensemble Temporal Fusion Transformer (TFT) models to make trading decisions based on price data and order book insights.

## Features

- **Ensemble TFT Models**: Combines directional and downward specialist models for improved prediction
- **Real-time Order Book Tracking**: Captures order book dynamics for feature enrichment
- **Multiple Ensemble Methods**: Supports different strategies for combining model predictions
- **Position Sizing**: Dynamically adjusts position sizes based on signal strength
- **Risk Management**: Implements stop-loss, take-profit, and max holding periods
- **Database Integration**: Stores predictions, decisions, and trades in TimescaleDB
- **Paper Trading**: Simulates trades for testing before enabling live trading
- **Docker Support**: Easily deployable via Docker containers

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- TimescaleDB (PostgreSQL with time-series extension)
- Access to Binance Futures API (for live trading)
- Pre-trained TFT models

## Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd realtime_tft_bot
   ```

2. **Environment Configuration**
   ```bash
   cp sample.env .env
   # Edit .env with your configuration
   ```

3. **Database Setup**
   ```bash
   # Create the necessary database schema
   cat ../database_setup/tftbot_schema_fixed.sql | docker exec -i <timescaledb-container> psql -U <username> -d <dbname>
   ```

4. **Model Files**
   ```bash
   # Place your trained TFT models in the models directory
   mkdir -p ../models
   # Copy your model files to ../models/
   ```

5. **Start the Bot (Docker)**
   ```bash
   docker-compose up -d
   ```

6. **Start the Bot (Local Development)**
   ```bash
   pip install -r requirements.txt
   python realtime_tft_bot.py
   ```

## Configuration Options

The bot is highly configurable through environment variables:

### Model Configuration
- `DIR_MODEL_PATH`: Path to directional TFT model
- `DOWN_MODEL_PATH`: Path to downward specialist TFT model
- `CONTEXT_LENGTH`: Number of time steps for model input

### Trading Logic
- `ENSEMBLE_METHOD`: How to combine model predictions (`selective`, `average`, `weighted`, or `extreme`)
- `TRADING_THRESHOLD`: Minimum prediction value to trigger a trade
- `STOP_LOSS`: Stop loss percentage (e.g., 0.002 for 0.2%)
- `TAKE_PROFIT`: Take profit percentage
- `MAX_HOLDING_PERIOD`: Maximum number of periods to hold a position

### Database & Trading Pair
- `TRADING_PAIR`: Trading pair to trade (e.g., WLD-USDT)
- `DB_HOST`, `DB_PORT`, etc.: Database connection details

### Execution
- `ENABLE_LIVE_TRADING`: Set to `true` to enable actual trading

## Database Schema

The bot uses several tables in TimescaleDB:
- `tft_predictions`: Stores model predictions
- `trade_decisions`: Records trading decisions
- `executed_trades`: Logs trade executions
- `order_events`: Tracks order book events
- `position_state`: Maintains current position information

## Monitoring & Logs

- Logs are written to `tft_bot.log` and stdout
- Detailed trade history and predictions are stored in the database

## Usage Examples

### Running in Simulation Mode
```bash
# Set ENABLE_LIVE_TRADING=false in .env
docker-compose up
```

### Enabling Live Trading
```bash
# Set up API keys and enable live trading
echo "BINANCE_API_KEY=your_api_key" >> .env
echo "BINANCE_API_SECRET=your_api_secret" >> .env
echo "ENABLE_LIVE_TRADING=true" >> .env
docker-compose up -d
```

### Querying Trade Performance
```sql
-- Example query to get recent trade performance
SELECT 
  decision, 
  COUNT(*) as count, 
  AVG(pnl) as avg_pnl 
FROM trade_decisions 
WHERE decision_timestamp > NOW() - INTERVAL '24 hours' 
GROUP BY decision;
```

## Safety Warning

This bot can perform real trades with real money. Always:
- Test thoroughly in simulation mode first
- Start with small position sizes
- Monitor the bot regularly
- Use secure API credentials

## License

[MIT License](LICENSE) 