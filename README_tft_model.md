# Improved Temporal Fusion Transformer (TFT) for Cryptocurrency Price Prediction

This repository contains an improved implementation of the Temporal Fusion Transformer (TFT) model for cryptocurrency price prediction based on high-frequency (10-second) bar data. The model leverages time series data from order books and market data to predict short-term price movements.

## Project Structure

```
.
├── models/                         # Model implementations and saved model files
│   ├── improved_tft.py             # Improved TFT model with variable selection networks
│   ├── temporal_fusion_transformer.py  # Original TFT implementation (for reference)
│   └── saved_models/               # Directory for saved model checkpoints
├── experiments/                    # Experiment tracking
│   └── tft_10sec_v0.1/             # Initial experiment results
│       ├── summary.md              # Experiment summary
│       └── images/                 # Visualization images
├── docs/                           # Documentation
│   └── tft_production_roadmap.md   # Production roadmap
├── train_improved_tft.py           # Training script
├── evaluate_improved_tft.py        # Evaluation script
├── serve_tft_model.py              # Model serving API
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Docker compose configuration
├── requirements.txt                # Python dependencies
└── README_tft_model.md             # This file
```

## Features

- **Full TFT Architecture**: Implementation of the complete Temporal Fusion Transformer architecture with variable selection networks, as described in the original paper.
- **Enhanced Feature Engineering**: Automatic detection and correction of data quality issues, derived features, and proper handling of time-based features.
- **Robust Training**: Early stopping, learning rate scheduling, and bias correction mechanisms.
- **Comprehensive Evaluation**: Tools for assessing model performance including directional accuracy, trading metrics, and visualization.
- **Production-Ready Serving**: REST API for model inference with automatic data refresh.
- **Containerization**: Docker setup for consistent deployment.
- **Monitoring**: Prometheus/Grafana integration for model monitoring.

## Prerequisites

- Python 3.9+
- PostgreSQL database with cryptocurrency market data
- CUDA-compatible GPU (recommended for training)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quants-lab.git
   cd quants-lab
   ```

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/quants-lab.git
   cd quants-lab
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables for database connection:
   ```bash
   export DB_HOST=localhost
   export DB_PORT=5438
   export DB_USER=backtest_user
   export DB_PASSWORD=backtest_password
   export DB_NAME=backtest_db
   ```

## Usage

### Training a Model

```bash
python train_improved_tft.py --days 14 --context 30 --batch 64 --hidden 128 --heads 4 --dropout 0.1 --lr 1e-4 --epochs 100
```

Options:
- `--days`: Number of days of data to use for training
- `--missing`: Strategy for handling missing values (`fill_zero`, `fill_mean`, `fill_forward`, `interpolate`)
- `--context`: Context length (number of time steps to use as input)
- `--batch`: Batch size
- `--hidden`: Hidden layer size
- `--heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--lr`: Learning rate
- `--epochs`: Maximum number of training epochs
- `--patience`: Early stopping patience
- `--cpu`: Force CPU training even if GPU is available

### Evaluating a Model

```bash
python evaluate_improved_tft.py --model models/improved_tft_20250501_230054.pt --days 7 --output evaluation_output
```

Options:
- `--model`: Path to the model file
- `--days`: Number of days of recent data for evaluation
- `--missing`: Strategy for handling missing values
- `--output`: Directory for saving evaluation results
- `--context`: Override the context length from model config

### Serving the Model

```bash
python serve_tft_model.py --host 0.0.0.0 --port 5000
```

Options:
- `--host`: Host address to bind
- `--port`: Port to bind
- `--debug`: Run in debug mode
- `--model`: Specific model file to load (otherwise loads latest model)

## API Endpoints

The model serving API provides the following endpoints:

- `GET /health`: Health check for the API
- `GET /predict`: Get a prediction for the next time step
- `POST /refresh`: Force refresh of the model and data
- `GET /config`: Get the current model configuration

Example API usage:

```python
import requests

# Get a prediction
response = requests.get("http://localhost:5000/predict")
result = response.json()
print(f"Prediction: {result['data']['prediction']}")

# Force refresh
requests.post("http://localhost:5000/refresh")
```

## Model Performance

Initial evaluation of the improved TFT model shows significant enhancements over the original implementation:

- Reduced Mean Absolute Error (MAE) by approximately 20%
- Improved directional accuracy from ~53% to ~57%
- Reduced prediction bias from -0.3097 to -0.0852
- Better Sharpe ratio in trading simulation

For detailed performance metrics, see the evaluation output or run the evaluation script on your own data.

## Monitoring

The Docker Compose setup includes Prometheus and Grafana for monitoring:

- **Prometheus**: Access at http://localhost:9090
- **Grafana**: Access at http://localhost:3000 (default credentials: admin/admin)

A pre-configured dashboard is available for monitoring model performance and server health.

## Development and Customization

### Adding New Features

To add new features to the model:

1. Modify the `add_derived_features` function in `train_improved_tft.py`
2. Ensure the new features are included in the data preparation process in both training and serving code
3. Retrain the model to incorporate the new features

### Customizing the Model Architecture

To customize the model architecture:

1. Modify the `TemporalFusionTransformer` class in `models/improved_tft.py`
2. Adjust hyperparameters through the training script arguments

## License

[MIT License](LICENSE)

## Acknowledgements

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) paper by Bryan Lim, et al.
- PyTorch and PyTorch Lightning teams for their excellent frameworks 