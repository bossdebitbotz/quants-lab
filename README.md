# Quants-Lab: TFT Cryptocurrency Price Prediction

## Overview

This project develops Temporal Fusion Transformer (TFT) models for high-frequency cryptocurrency price prediction. We're implementing and refining machine learning models to capture market dynamics and forecast short-term price movements.

## Project Structure

```
quants-lab/
├── .cursorrules          # Project development guidelines
├── docs/                 # Documentation
│   ├── changelog.md      # Version history and progress
│   ├── features.md       # Feature definitions and engineering
│   ├── model_architecture.md # TFT architecture details
│   ├── evaluation_metrics.md # Evaluation methodology
│   └── training_protocol.md  # Standard training procedures
├── models/               # Saved model files
│   └── tft_10sec_bias_corrected_20250501_230054.pt # Latest model
├── experiments/          # Experiment tracking
│   └── tft_10sec_v0.1/   # Current experiment
│       ├── config.json   # Configuration parameters
│       ├── summary.md    # Experiment results
│       └── images/       # Visualizations and plots
└── evaluate_tft_10sec_model.ipynb # Evaluation notebook
```

## Current Status

We're currently at version 0.1.0 of our TFT model for 10-second bar data. Key characteristics:

- 21 time-varying features (price, volume, order book metrics)
- 128-dimensional hidden layers with 4 attention heads
- Context window of 30 time steps (5 minutes of market data)
- Bias correction to address systematic prediction errors

See `docs/changelog.md` for detailed version history and progress.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- PostgreSQL database with market data
- Required packages: numpy, pandas, matplotlib, seaborn, psycopg2

### Setup

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Configure database connection in evaluation notebook
4. Run evaluation notebook to test models

### Database Configuration

The project uses a PostgreSQL database with the following configuration:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}
```

## Model Architecture

We're implementing a modified version of the Temporal Fusion Transformer as described in the paper by Lim et al. (2019). See `docs/model_architecture.md` for details on our implementation.

## Development Workflow

1. Follow the `docs/training_protocol.md` for training new models
2. Document experiments in the `experiments/` directory
3. Evaluate models using standard metrics in `docs/evaluation_metrics.md`
4. Update changelog with progress and version history

## Future Development

Key areas for improvement:

1. Enhanced feature engineering (market regimes, advanced order book features)
2. Full TFT architecture implementation with variable selection networks
3. Robust trading simulation and strategy development
4. Cross-validation across different market conditions

See the [Current Experiment Summary](experiments/tft_10sec_v0.1/summary.md) for specific next steps.

## Contributing

1. Follow the guidelines in `.cursorrules`
2. Document all changes and experiments thoroughly
3. Maintain model reproducibility at all times
4. Ensure all code has appropriate tests

## License

This project is proprietary and confidential.

## Acknowledgments

- Based on the paper ["Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"](https://arxiv.org/abs/1912.09363) by Bryan Lim et al.
- Inspired by PyTorch Forecasting and GluonTS implementations of TFT
