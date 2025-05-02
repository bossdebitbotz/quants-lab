# Feature Documentation for TFT Models

## Overview

This document describes the features used in our TFT models for cryptocurrency price prediction. We currently work with 10-second bar data, extracting various price, volume, order book, and derived features to capture market dynamics.

## Feature Categories

### Price Features

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `open` | Opening price of the bar | Raw | Provides price level information |
| `high` | Highest price during the bar | Raw | Captures price range/volatility |
| `low` | Lowest price during the bar | Raw | Captures price range/volatility |
| `close` | Closing price of the bar | Raw | Most recent price information |

### Volume Features

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `volume` | Total trading volume | Raw | Measures trading activity |
| `bid_vol` | Volume at bid | Raw | Buy-side pressure |
| `ask_vol` | Volume at ask | Raw | Sell-side pressure |

### Order Book Features

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `imbalance` | Order book imbalance | (bid_vol - ask_vol) / (bid_vol + ask_vol) | Market directional pressure |
| `spread_mean` | Average bid-ask spread | Raw | Liquidity measure |
| `spread_pct_mean` | Spread as percentage of price | spread_mean / mid_price | Normalized liquidity |

### Order Flow Features

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `new_bid_orders` | New buy orders added | Raw count | Buy-side interest |
| `new_ask_orders` | New sell orders added | Raw count | Sell-side interest |
| `canceled_bid_orders` | Canceled buy orders | Raw count | Buy-side sentiment change |
| `canceled_ask_orders` | Canceled sell orders | Raw count | Sell-side sentiment change |
| `executed_bid_orders` | Executed buy orders | Raw count | Actual buy transactions |
| `executed_ask_orders` | Executed sell orders | Raw count | Actual sell transactions |
| `buy_sell_imbalance` | Order flow imbalance | (executed_bid - executed_ask) / (executed_bid + executed_ask) | Net pressure |

### Technical Indicators

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `returns_10sec` | 10-second price returns | log(close / prev_close) | Recent price movement |
| `returns_30sec` | 30-second price returns | log(close / close_3periods_ago) | Medium-term momentum |
| `returns_1min` | 1-minute price returns | log(close / close_6periods_ago) | Longer-term momentum |
| `volatility_1min` | 1-minute rolling volatility | std(returns_10sec, window=6) | Recent market volatility |

### Target Variable

| Feature | Description | Engineering | Rationale |
|---------|-------------|-------------|-----------|
| `next_return` | Next 10-second return | log(next_close / close) | Prediction target |

## Data Preprocessing

1. **Normalization**:
   - All features are z-score normalized: (x - mean) / std
   - Statistics are calculated on the training set
   - Features with zero std dev are set to zero

2. **Missing Value Handling**:
   - All NaN values are filled with zeros after normalization
   - This approach preserves the mean value for missing data

3. **Temporal Structure**:
   - Context window of 30 time steps (5 minutes of market data)
   - Each prediction uses the past 30 bars to predict the next return

## Feature Engineering Considerations

1. **Lookahead Prevention**:
   - All features are calculated using strictly past or current information
   - Rolling windows only use past values
   - No future information leakage in any calculation

2. **Feature Selection Criteria**:
   - Theoretical relevance to price formation
   - Empirical predictive power in preliminary analysis
   - Low collinearity with other features
   - Signal-to-noise ratio

## Known Issues and Limitations

1. Several features contain missing values in the current dataset:
   - `volume`, `bid_vol`, `ask_vol`, `spread_mean` often contain NaNs
   - `executed_bid_orders`, `executed_ask_orders`, `buy_sell_imbalance` may be incomplete

2. The `returns_*` and `volatility_*` features show zero standard deviation in the evaluation dataset, suggesting potential data quality issues.

## Future Feature Development

1. **Market Regime Indicators**:
   - Add trend/mean-reversion regime classification
   - Include volatility regime indicators

2. **Advanced Order Book Features**:
   - Order book depth metrics at multiple levels
   - Volume-weighted price levels
   - Order book pressure imbalance

3. **Cross-Exchange Indicators**:
   - Price spreads between exchanges
   - Volume imbalances across venues
   - Arbitrage opportunity metrics

4. **Sentiment Features**:
   - NLP-derived sentiment scores
   - Social media metrics
   - News event indicators 