import nbformat as nbf
import os

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Title and introduction
title_md = """
# TFT Model Evaluation Notebook

This notebook loads our trained Temporal Fusion Transformer model, makes predictions on test data, and visualizes the predicted vs actual price movements.
"""

# Import statements
imports_code = """
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from datetime import datetime, timedelta
from models.temporal_fusion_transformer import TemporalFusionTransformer, create_data_loaders

# Configure plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
"""

# Database connection
db_section_md = "## 1. Database Connection & Data Loading"
db_conn_code = """
# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def get_db_connection():
    # Create a connection to the PostgreSQL database
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None
"""

# Data loading
data_loading_code = """
# Load data from the database
conn = get_db_connection()

query = \"\"\"
SELECT 
    timestamp,
    mid_price,
    spread,
    spread_pct,
    imbalance,
    new_bid_orders,
    new_ask_orders,
    canceled_bid_orders,
    canceled_ask_orders,
    next_price_1min,
    fill_probability
FROM tft_features
ORDER BY timestamp
\"\"\"

df = pd.read_sql_query(query, conn)
conn.close()

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Calculate target (price change)
df['target'] = df['next_price_1min'] / df['mid_price'] - 1

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
df.head()
"""

# Data preparation
data_prep_md = "## 2. Data Preparation"
data_prep_code = """
# Define feature columns
feature_columns = [
    'spread',
    'spread_pct',
    'imbalance',
    'new_bid_orders',
    'new_ask_orders',
    'canceled_bid_orders',
    'canceled_ask_orders',
    'fill_probability'
]

# Normalize features (same as in training script)
df_scaled = df.copy()
for col in feature_columns:
    mean = df[col].mean()
    std = df[col].std()
    df_scaled[col] = (df[col] - mean) / (std if std > 0 else 1)

# Split into train, validation, and test sets (use same ratios as training script)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_size = int(len(df_scaled) * train_ratio)
val_size = int(len(df_scaled) * val_ratio)

train_data = df_scaled.iloc[:train_size]
val_data = df_scaled.iloc[train_size:train_size + val_size]
test_data = df_scaled.iloc[train_size + val_size:]

print(f"Train set: {len(train_data)} samples")
print(f"Validation set: {len(val_data)} samples")
print(f"Test set: {len(test_data)} samples")
"""

# Load trained model
model_md = "## 3. Load Trained Model"
model_code = """
# Find the latest model file
model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.startswith('tft_model_') and f.endswith('.pt')]
latest_model = sorted(model_files)[-1]  # Get the most recent model
model_path = os.path.join(model_dir, latest_model)

print(f"Loading model from {model_path}")

# Initialize model with the same architecture as during training
model = TemporalFusionTransformer(
    time_varying_features=10,  # Number of features in our dataset
    hidden_size=64,
    num_heads=4,
    dropout=0.1,
    learning_rate=1e-3,
    context_length=10,  # Using 10 minutes of history
    prediction_length=1  # Predicting 1 minute ahead
)

# Load the saved weights
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode
"""

# Create dataloader
dataloader_md = "## 4. Create DataLoaders for Model Evaluation"
dataloader_code = """
# Create data loaders with same parameters as training
context_length = 10
prediction_length = 1
batch_size = 8

_, _, test_loader = create_data_loaders(
    data=df_scaled,
    context_length=context_length,
    prediction_length=prediction_length,
    batch_size=batch_size,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    target_column='target'
)
"""

# Generate predictions
predictions_md = "## 5. Generate Predictions on Test Set"
predictions_code = """
# Collect test predictions
actual_values = []
predicted_values = []
timestamps = []

with torch.no_grad():
    for batch in test_loader:
        # Get input features and target values
        temporal_features = batch['temporal_features']
        targets = batch['targets'].squeeze(-1)
        
        # Make predictions
        predictions = model(None, temporal_features).squeeze(-1)
        
        # Collect results
        actual_values.extend(targets.cpu().numpy())
        predicted_values.extend(predictions.cpu().numpy())

# Create a DataFrame with predictions
test_indices = test_data.index[-len(actual_values):]
results_df = pd.DataFrame({
    'timestamp': test_indices,
    'actual_return': actual_values,
    'predicted_return': predicted_values
})
results_df.set_index('timestamp', inplace=True)

# Calculate evaluation metrics
mae = np.mean(np.abs(results_df['actual_return'] - results_df['predicted_return']))
mse = np.mean((results_df['actual_return'] - results_df['predicted_return'])**2)
rmse = np.sqrt(mse)
correlation = np.corrcoef(results_df['actual_return'], results_df['predicted_return'])[0, 1]

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Correlation: {correlation:.6f}")

# Display first few predictions
results_df.head()
"""

# Visualizations
vis_md = "## 6. Visualize Predictions vs Actual Values"
vis_code1 = """
# Plot predictions vs actual values over time
plt.figure(figsize=(16, 8))
plt.plot(results_df.index, results_df['actual_return'], label='Actual Returns', color='blue', alpha=0.7)
plt.plot(results_df.index, results_df['predicted_return'], label='Predicted Returns', color='red', alpha=0.7)
plt.title('Actual vs Predicted Returns Over Time', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Returns (% change)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

vis_code2 = """
# Zoom in on a specific time window (last 100 predictions)
zoom_window = results_df.iloc[-100:]

plt.figure(figsize=(16, 8))
plt.plot(zoom_window.index, zoom_window['actual_return'], label='Actual Returns', color='blue', marker='o', markersize=4, alpha=0.7)
plt.plot(zoom_window.index, zoom_window['predicted_return'], label='Predicted Returns', color='red', marker='x', markersize=4, alpha=0.7)
plt.title('Zoomed View: Actual vs Predicted Returns (Last 100 Minutes)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Returns (% change)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

vis_code3 = """
# Scatter plot of predicted vs actual returns
plt.figure(figsize=(10, 10))
plt.scatter(results_df['actual_return'], results_df['predicted_return'], alpha=0.5)
plt.plot([-0.01, 0.01], [-0.01, 0.01], 'r--') # Diagonal line for perfect predictions
plt.title('Predicted vs Actual Returns', fontsize=16)
plt.xlabel('Actual Returns', fontsize=14)
plt.ylabel('Predicted Returns', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

# Error analysis
error_md = "## 7. Analyze Prediction Errors"
error_code1 = """
# Calculate prediction errors
results_df['error'] = results_df['actual_return'] - results_df['predicted_return']
results_df['abs_error'] = np.abs(results_df['error'])

# Plot error distribution
plt.figure(figsize=(12, 8))
sns.histplot(results_df['error'], kde=True, bins=50)
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of Prediction Errors', fontsize=16)
plt.xlabel('Prediction Error', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

error_code2 = """
# Plot errors over time to look for patterns
plt.figure(figsize=(16, 8))
plt.plot(results_df.index, results_df['error'], alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Prediction Errors Over Time', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

# Trading simulation
trading_md = "## 8. Analyze Trading Performance"
trading_code1 = """
# Simulate a simple trading strategy based on predictions
def simulate_trading(predictions_df, threshold=0.0001):
    # Simulate a simple trading strategy based on model predictions
    df = predictions_df.copy()
    
    # Generate trading signals
    df['signal'] = 0  # 0: no position, 1: long, -1: short
    df.loc[df['predicted_return'] > threshold, 'signal'] = 1  # Buy if predicted return exceeds threshold
    df.loc[df['predicted_return'] < -threshold, 'signal'] = -1  # Sell if predicted return is below negative threshold
    
    # Calculate strategy returns (assuming we hold for exactly 1 minute)
    df['strategy_return'] = df['signal'] * df['actual_return']
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1
    df['cumulative_market'] = (1 + df['actual_return']).cumprod() - 1
    
    # Calculate trading statistics
    total_trades = (df['signal'] != df['signal'].shift(1)).sum()
    winning_trades = (df['strategy_return'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate annualized returns and Sharpe ratio (assuming 1-minute bars, 12 trading hours per day, 250 trading days per year)
    total_minutes = len(df)
    annual_factor = (250 * 12 * 60) / total_minutes
    strategy_return = df['cumulative_return'].iloc[-1]
    market_return = df['cumulative_market'].iloc[-1]
    
    annualized_return = (1 + strategy_return) ** annual_factor - 1
    annualized_market = (1 + market_return) ** annual_factor - 1
    
    # Calculate volatility and Sharpe ratio (assuming risk-free rate of 0%)
    daily_returns = df['strategy_return'].dropna().resample('1D').sum()
    annualized_vol = daily_returns.std() * np.sqrt(250)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + df['strategy_return']).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak - 1)
    max_drawdown = drawdown.min()
    
    # Print trading statistics
    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades} ({win_rate:.2%})")
    print(f"Total return: {strategy_return:.2%} (Market: {market_return:.2%})")
    print(f"Annualized return: {annualized_return:.2%} (Market: {annualized_market:.2%})")
    print(f"Annualized volatility: {annualized_vol:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")
    
    return df

# Run trading simulation
trading_results = simulate_trading(results_df, threshold=0.0001)
"""

trading_code2 = """
# Plot trading strategy performance
plt.figure(figsize=(16, 8))
plt.plot(trading_results.index, trading_results['cumulative_return'], label='Strategy', color='green')
plt.plot(trading_results.index, trading_results['cumulative_market'], label='Buy & Hold', color='blue')
plt.title('Trading Strategy Performance', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Return', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

# Feature importance
feature_md = "## 9. Analyze Feature Importance"
feature_code = """
# Calculate feature correlation with prediction accuracy
# Get the original features for correlation analysis
feature_importance = pd.DataFrame(index=feature_columns)

for feature in feature_columns:
    # Extract feature values for the test period
    feature_values = test_data[feature].values[-len(results_df):]
    
    # Calculate correlation between feature and prediction accuracy
    correlation = np.corrcoef(feature_values, results_df['abs_error'])[0, 1]
    feature_importance.loc[feature, 'error_correlation'] = correlation

# Sort by absolute correlation
feature_importance['abs_correlation'] = np.abs(feature_importance['error_correlation'])
feature_importance = feature_importance.sort_values('abs_correlation', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance.index, feature_importance['abs_correlation'], color='teal')
plt.title('Feature Importance (Correlation with Prediction Error)', fontsize=16)
plt.xlabel('Absolute Correlation', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

# Conclusion
conclusion_md = """
## 10. Conclusion & Next Steps

Based on the visualizations and analysis above, here are some potential next steps:

1. **Feature Engineering**: Focus on improving the features that show the highest correlation with prediction errors
2. **Model Tuning**: Experiment with different architectures, especially attention mechanisms
3. **Trading Strategy**: Refine the trading strategy by optimizing entry/exit thresholds
4. **Additional Data**: Consider incorporating more data sources (e.g., trade volume, market sentiment)
5. **Ensemble Models**: Try combining the TFT model with other approaches for more robust predictions
"""

# Combine all cells
cells = [
    nbf.v4.new_markdown_cell(title_md),
    nbf.v4.new_code_cell(imports_code),
    nbf.v4.new_markdown_cell(db_section_md),
    nbf.v4.new_code_cell(db_conn_code),
    nbf.v4.new_code_cell(data_loading_code),
    nbf.v4.new_markdown_cell(data_prep_md),
    nbf.v4.new_code_cell(data_prep_code),
    nbf.v4.new_markdown_cell(model_md),
    nbf.v4.new_code_cell(model_code),
    nbf.v4.new_markdown_cell(dataloader_md),
    nbf.v4.new_code_cell(dataloader_code),
    nbf.v4.new_markdown_cell(predictions_md),
    nbf.v4.new_code_cell(predictions_code),
    nbf.v4.new_markdown_cell(vis_md),
    nbf.v4.new_code_cell(vis_code1),
    nbf.v4.new_code_cell(vis_code2),
    nbf.v4.new_code_cell(vis_code3),
    nbf.v4.new_markdown_cell(error_md),
    nbf.v4.new_code_cell(error_code1),
    nbf.v4.new_code_cell(error_code2),
    nbf.v4.new_markdown_cell(trading_md),
    nbf.v4.new_code_cell(trading_code1),
    nbf.v4.new_code_cell(trading_code2),
    nbf.v4.new_markdown_cell(feature_md),
    nbf.v4.new_code_cell(feature_code),
    nbf.v4.new_markdown_cell(conclusion_md)
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to disk
with open('evaluate_tft_model.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully: evaluate_tft_model.ipynb") 