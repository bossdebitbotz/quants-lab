import os
import logging
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
import psycopg2
import torch
from models.temporal_fusion_transformer import TemporalFusionTransformer, create_data_loaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_10sec_trainer")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
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
        logger.error(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def load_training_data():
    """Load and prepare 10-second bar training data from the database"""
    try:
        conn = get_db_connection()
        
        # Get the latest date in the database
        date_query = """
        SELECT MAX(timestamp) FROM tft_features_10sec
        """
        with conn.cursor() as cursor:
            cursor.execute(date_query)
            latest_date = cursor.fetchone()[0]
        
        if latest_date is None:
            logger.error("No data found in tft_features_10sec table")
            return None, None
        
        # Calculate the date 7 days ago from the latest date
        start_date = latest_date - timedelta(days=7)
        
        logger.info(f"Fetching data from {start_date} to {latest_date}")
        
        # Query to get all features from 10-second bars for the last 7 days
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            COALESCE(volume, 0) as volume,
            COALESCE(bid_vol, 0) as bid_vol,
            COALESCE(ask_vol, 0) as ask_vol,
            COALESCE(imbalance, 0) as imbalance,
            COALESCE(spread_mean, 0) as spread_mean,
            COALESCE(spread_pct_mean, 0) as spread_pct_mean,
            COALESCE(new_bid_orders, 0) as new_bid_orders,
            COALESCE(new_ask_orders, 0) as new_ask_orders,
            COALESCE(canceled_bid_orders, 0) as canceled_bid_orders,
            COALESCE(canceled_ask_orders, 0) as canceled_ask_orders,
            COALESCE(executed_bid_orders, 0) as executed_bid_orders,
            COALESCE(executed_ask_orders, 0) as executed_ask_orders,
            COALESCE(buy_sell_imbalance, 0) as buy_sell_imbalance,
            COALESCE(returns_10sec, 0) as returns_10sec,
            COALESCE(returns_30sec, 0) as returns_30sec,
            COALESCE(returns_1min, 0) as returns_1min,
            COALESCE(volatility_1min, 0) as volatility_1min,
            COALESCE(target_10sec, 0) as target_10sec
        FROM tft_features_10sec
        WHERE timestamp >= %s
        ORDER BY timestamp
        """
        
        # Read data into DataFrame
        df = pd.read_sql_query(query, conn, params=[start_date])
        
        logger.info(f"Retrieved {len(df)} rows of raw data")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values in columns: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Define feature columns (excluding timestamp and target-related columns)
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'bid_vol', 'ask_vol', 'imbalance',
            'spread_mean', 'spread_pct_mean',
            'new_bid_orders', 'new_ask_orders',
            'canceled_bid_orders', 'canceled_ask_orders',
            'executed_bid_orders', 'executed_ask_orders',
            'buy_sell_imbalance',
            'returns_10sec', 'returns_30sec', 'returns_1min',
            'volatility_1min'
        ]
        
        # Ensure all features are numeric
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any remaining NaN values with 0 instead of dropping rows
        df = df.fillna(0)
        
        # Log data statistics before normalization
        logger.info(f"Data summary before normalization:")
        for col in feature_columns:
            if col in df.columns:
                logger.info(f"{col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}, std={df[col].std():.6f}")
        
        # Normalize features
        df_scaled = df.copy()
        for col in feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            # Avoid division by zero
            if std == 0:
                logger.warning(f"Standard deviation for {col} is 0, skipping normalization")
                df_scaled[col] = 0
            else:
                df_scaled[col] = (df[col] - mean) / std
        
        # Check if all features are valid (no inf or nan)
        invalid_features = df_scaled.isin([np.inf, -np.inf, np.nan]).any(axis=0)
        if invalid_features.any():
            logger.warning(f"Invalid values in columns after normalization: {invalid_features[invalid_features].index.tolist()}")
            # Replace inf values with 0
            df_scaled = df_scaled.replace([np.inf, -np.inf], 0)
        
        # Rename target column to 'target' for consistency with the model
        df_scaled['target'] = df_scaled['target_10sec']
        
        logger.info(f"Processed {len(df_scaled)} samples with {len(feature_columns)} features")
        logger.info(f"Time span: {df_scaled.index[0]} to {df_scaled.index[-1]}")
        
        return df_scaled, feature_columns
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    finally:
        if conn:
            conn.close()

def train_model(
    train_loader,
    val_loader,
    test_loader,
    num_features,
    max_epochs=100
):
    """Train the TFT model on 10-second bar data"""
    
    logger.info(f"Initializing model with {num_features} features")
    
    # Initialize model with appropriate settings for 10-second data
    model = TemporalFusionTransformer(
        time_varying_features=num_features,  # Explicitly set to match our actual feature count
        hidden_size=128,  # Increased from 64 to capture more complex patterns
        num_heads=4,
        dropout=0.2,  # Increased dropout to prevent overfitting on higher frequency data
        learning_rate=5e-4,  # Lower learning rate for more stable training
        context_length=30,  # 30 x 10-second = 5 minutes of history
        prediction_length=1,  # Predicting 1 10-second interval ahead
        bias_correction=True  # Enable bias correction during training
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/10sec',
        filename='tft-10sec-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience for higher frequency data
        mode='min'
    )
    
    # Create directory if it doesn't exist
    os.makedirs('checkpoints/10sec', exist_ok=True)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',  # Change to 'gpu' if available
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=True
    )
    
    # Train model
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Test model
        trainer.test(model, dataloaders=test_loader)
        
        # Check for prediction bias after training
        bias = check_prediction_bias(model, test_loader)
        logger.info(f"Model prediction bias: {bias:.6f}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to save the model even if training fails
        model_path = f"models/tft_10sec_model_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Partially trained model saved to {model_path}")
        except:
            logger.error("Could not save partial model")
    
    # Save final model
    final_model_path = f"models/tft_10sec_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Trained model saved to {final_model_path}")
    
    return model, trainer

def check_prediction_bias(model, dataloader):
    """Check for systematic bias in the model predictions"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            temporal_features = batch['temporal_features']
            targets = batch['targets'].squeeze(-1)
            
            # Ensure correct feature count
            if temporal_features.shape[2] > model.time_varying_features:
                temporal_features = temporal_features[:, :, :model.time_varying_features]
            
            # Make predictions
            predictions = model(None, temporal_features).squeeze(-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average prediction bias
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate bias as mean prediction error
    bias = np.mean(all_preds - all_targets)
    return bias

def main():
    logger.info("Starting TFT model training on 10-second bars")
    
    # Load data
    df, feature_columns = load_training_data()
    if df is None or feature_columns is None:
        logger.error("Failed to load training data")
        return
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_columns)} features")
    
    # Make sure to drop any columns that shouldn't be used as features
    # Columns to exclude from features (target and any metadata columns)
    exclude_columns = ['target_10sec', 'target', 'target_30sec', 'target_1min']
    
    # Filter feature columns to ensure we use only the ones we want
    filtered_features = [col for col in feature_columns if col in df.columns and col not in exclude_columns]
    
    logger.info(f"Using {len(filtered_features)} filtered features: {filtered_features}")
    
    # Create data loaders with appropriate context length for 10-second data
    train_loader, val_loader, test_loader = create_data_loaders(
        data=df[filtered_features + ['target']],  # Include only filtered features and target column
        context_length=30,  # 30 x 10-second = 5 minutes of history
        prediction_length=1,  # 1 x 10-second prediction
        batch_size=32,  # Reduced batch size due to potentially larger dataset
        train_ratio=0.7,
        val_ratio=0.15,
        target_column='target'
    )
    
    # Train model
    model, trainer = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=len(filtered_features),
        max_epochs=100  # Increased epochs for better convergence
    )
    
    logger.info("Model training completed")

if __name__ == "__main__":
    main() 