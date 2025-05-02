import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e2c04557",
   "metadata": {},
   "source": [
    "# TFT Model Evaluation Notebook\n",
    "\n",
    "This notebook loads our trained Temporal Fusion Transformer model, makes predictions on test data, and visualizes the predicted vs actual price movements."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "afc2c090",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import psycopg2\n",
    "from datetime import datetime, timedelta\n",
    "from models.temporal_fusion_transformer import TemporalFusionTransformer, MarketDataset\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "# Configure plots\n",
    "plt.style.use('ggplot')\n",
    "sns.set(font_scale=1.2)\n",
    "plt.rcParams['figure.figsize'] = [12, 8]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8266c64",
   "metadata": {},
   "source": [
    "## 1. Database Connection & Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "2fb239f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Database configuration\n",
    "DB_CONFIG = {\n",
    "    'host': 'localhost',\n",
    "    'port': 5438,\n",
    "    'user': 'backtest_user',\n",
    "    'password': 'backtest_password',\n",
    "    'database': 'backtest_db'\n",
    "}\n",
    "\n",
    "def get_db_connection():\n",
    "    # Create a connection to the PostgreSQL database\n",
    "    try:\n",
    "        conn = psycopg2.connect(\n",
    "            host=DB_CONFIG['host'],\n",
    "            port=DB_CONFIG['port'],\n",
    "            user=DB_CONFIG['user'],\n",
    "            password=DB_CONFIG['password'],\n",
    "            database=DB_CONFIG['database']\n",
    "        )\n",
    "        return conn\n",
    "    except Exception as e:\n",
    "        print(f\"Error connecting to database: {type(e).__name__} - {str(e)}\")\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ed398a00",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Connect to database and load data\n",
    "conn = get_db_connection()\n",
    "\n",
    "# Query to get features\n",
    "query = \"\"\"\n",
    "SELECT \n",
    "    timestamp,\n",
    "    mid_price,\n",
    "    spread,\n",
    "    spread_pct,\n",
    "    imbalance,\n",
    "    new_bid_orders,\n",
    "    new_ask_orders,\n",
    "    canceled_bid_orders,\n",
    "    canceled_ask_orders,\n",
    "    next_price_1min,\n",
    "    fill_probability\n",
    "FROM tft_features\n",
    "ORDER BY timestamp\n",
    "\"\"\"\n",
    "\n",
    "df = pd.read_sql_query(query, conn)\n",
    "conn.close()\n",
    "\n",
    "# Set timestamp as index\n",
    "df.set_index('timestamp', inplace=True)\n",
    "\n",
    "# Calculate target (price change)\n",
    "df['target'] = df['next_price_1min'] / df['mid_price'] - 1\n",
    "\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "print(f\"Date range: {df.index[0]} to {df.index[-1]}\")\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a5412c0",
   "metadata": {},
   "source": [
    "## 2. Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "b23c5c07",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define feature columns\n",
    "feature_columns = [\n",
    "    'spread',\n",
    "    'spread_pct',\n",
    "    'imbalance',\n",
    "    'new_bid_orders',\n",
    "    'new_ask_orders',\n",
    "    'canceled_bid_orders',\n",
    "    'canceled_ask_orders',\n",
    "    'fill_probability'\n",
    "]\n",
    "\n",
    "# Normalize features (same as in training script)\n",
    "df_scaled = df.copy()\n",
    "for col in feature_columns:\n",
    "    mean = df[col].mean()\n",
    "    std = df[col].std()\n",
    "    df_scaled[col] = (df[col] - mean) / (std if std > 0 else 1)\n",
    "\n",
    "# Split into train, validation, and test sets (use same ratios as training script)\n",
    "train_ratio = 0.7\n",
    "val_ratio = 0.15\n",
    "test_ratio = 0.15\n",
    "\n",
    "train_size = int(len(df_scaled) * train_ratio)\n",
    "val_size = int(len(df_scaled) * val_ratio)\n",
    "\n",
    "train_data = df_scaled.iloc[:train_size]\n",
    "val_data = df_scaled.iloc[train_size:train_size + val_size]\n",
    "test_data = df_scaled.iloc[train_size + val_size:]\n",
    "\n",
    "print(f\"Train set: {len(train_data)} samples\")\n",
    "print(f\"Validation set: {len(val_data)} samples\")\n",
    "print(f\"Test set: {len(test_data)} samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b86c40e0",
   "metadata": {},
   "source": [
    "## 3. Load the Trained Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "88ea71e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find the latest checkpoint\n",
    "checkpoint_dir = 'checkpoints'\n",
    "checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]\n",
    "checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))\n",
    "\n",
    "latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1]) if checkpoint_files else None\n",
    "print(f\"Loading model from checkpoint: {latest_checkpoint}\")\n",
    "\n",
    "# Initialize and load model\n",
    "if latest_checkpoint:\n",
    "    # Use classmethod to load from checkpoint\n",
    "    model = TemporalFusionTransformer.load_from_checkpoint(latest_checkpoint)\n",
    "else:\n",
    "    # Initialize model with parameters if no checkpoint\n",
    "    model = TemporalFusionTransformer(\n",
    "        time_varying_features=8,  # Number of features in dataset\n",
    "        hidden_size=64,\n",
    "        num_heads=4,\n",
    "        dropout=0.1,\n",
    "        learning_rate=1e-3,\n",
    "        context_length=10,  # 10 minutes of history\n",
    "        prediction_length=1  # Predicting 1 minute ahead\n",
    "    )\n",
    "    print(\"No checkpoint found. Using untrained model.\")\n",
    "\n",
    "# Set model to evaluation mode\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12fb2ebe",
   "metadata": {},
   "source": [
    "## 4. Create Data Loaders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f04d6ca9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define collate function for batching\n",
    "def collate_fn(batch):\n",
    "    \"\"\"Custom collate function to handle batching of temporal sequences.\"\"\"\n",
    "    # Stack all temporal features\n",
    "    temporal_features = torch.stack([item['temporal_features'] for item in batch], dim=0)\n",
    "    targets = torch.stack([item['targets'] for item in batch], dim=0)\n",
    "    \n",
    "    output = {\n",
    "        'temporal_features': temporal_features,\n",
    "        'targets': targets\n",
    "    }\n",
    "    \n",
    "    # Handle static features if they exist\n",
    "    if 'static_features' in batch[0]:\n",
    "        static_features = torch.stack([item['static_features'] for item in batch], dim=0)\n",
    "        output['static_features'] = static_features\n",
    "    \n",
    "    return output\n",
    "\n",
    "# Create datasets\n",
    "batch_size = 64\n",
    "sequence_length = 10\n",
    "\n",
    "# Create test dataset\n",
    "test_dataset = MarketDataset(\n",
    "    test_data, \n",
    "    context_length=sequence_length,\n",
    "    prediction_length=1,\n",
    "    target_column='target'\n",
    ")\n",
    "\n",
    "# Create test dataloader\n",
    "test_loader = DataLoader(\n",
    "    test_dataset, \n",
    "    batch_size=batch_size, \n",
    "    shuffle=False,\n",
    "    collate_fn=collate_fn\n",
    ")\n",
    "\n",
    "print(f\"Number of batches in test loader: {len(test_loader)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc45aa19",
   "metadata": {},
   "source": [
    "## 5. Generate Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9a70ff48",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Collect test predictions\n",
    "actual_values = []\n",
    "predicted_values = []\n",
    "timestamps = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch in test_loader:\n",
    "        # Get input features and target values\n",
    "        temporal_features = batch['temporal_features']\n",
    "        targets = batch['targets'].squeeze(-1)\n",
    "        \n",
    "        # Make predictions\n",
    "        predictions = model(None, temporal_features).squeeze(-1)\n",
    "        \n",
    "        # Collect results\n",
    "        actual_values.extend(targets.cpu().numpy())\n",
    "        predicted_values.extend(predictions.cpu().numpy())\n",
    "\n",
    "# Convert lists to numpy arrays for proper calculations\n",
    "actual_values = np.array(actual_values)\n",
    "predicted_values = np.array(predicted_values)\n",
    "\n",
    "# Ensure arrays are 1-dimensional\n",
    "if len(actual_values.shape) > 1:\n",
    "    actual_values = actual_values.flatten()\n",
    "if len(predicted_values.shape) > 1:\n",
    "    predicted_values = predicted_values.flatten()\n",
    "    \n",
    "print(f\"Shape of actual values: {actual_values.shape}\")\n",
    "print(f\"Shape of predicted values: {predicted_values.shape}\")\n",
    "\n",
    "# Create a DataFrame with predictions\n",
    "test_indices = test_data.index[-len(actual_values):]\n",
    "results_df = pd.DataFrame({\n",
    "    'timestamp': test_indices,\n",
    "    'actual_return': actual_values,\n",
    "    'predicted_return': predicted_values\n",
    "})\n",
    "results_df.set_index('timestamp', inplace=True)\n",
    "\n",
    "# Calculate evaluation metrics\n",
    "mae = np.mean(np.abs(results_df['actual_return'] - results_df['predicted_return']))\n",
    "mse = np.mean((results_df['actual_return'] - results_df['predicted_return'])**2)\n",
    "rmse = np.sqrt(mse)\n",
    "\n",
    "# Safely calculate correlation\n",
    "if len(actual_values) > 0 and len(predicted_values) > 0:\n",
    "    correlation = np.corrcoef(actual_values, predicted_values)[0, 1]\n",
    "    print(f\"Correlation: {correlation:.6f}\")\n",
    "else:\n",
    "    print(\"Warning: No predictions generated, cannot calculate correlation\")\n",
    "\n",
    "print(f\"Mean Absolute Error (MAE): {mae:.6f}\")\n",
    "print(f\"Root Mean Squared Error (RMSE): {rmse:.6f}\")\n",
    "\n",
    "# Show samples of results\n",
    "results_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "caa32e18",
   "metadata": {},
   "source": [
    "## 6. Visualize Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "21f52eba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Time series of predictions versus actual values\n",
    "plt.figure(figsize=(16, 8))\n",
    "plt.plot(results_df.index, results_df['actual_return'], label='Actual Returns', alpha=0.7)\n",
    "plt.plot(results_df.index, results_df['predicted_return'], label='Predicted Returns', alpha=0.7)\n",
    "plt.title('Actual vs Predicted Returns Over Time')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Return')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "31f52eba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scatter plot of predicted vs actual\n",
    "plt.figure(figsize=(12, 12))\n",
    "plt.scatter(results_df['actual_return'], results_df['predicted_return'], alpha=0.5)\n",
    "plt.plot([-0.01, 0.01], [-0.01, 0.01], 'r--', alpha=0.7)\n",
    "plt.title('Predicted vs Actual Returns')\n",
    "plt.xlabel('Actual Return')\n",
    "plt.ylabel('Predicted Return')\n",
    "plt.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (Quants)",
   "language": "python",
   "name": "quants-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Write the notebook to a new file
with open('fixed_evaluate_tft_model.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!") 