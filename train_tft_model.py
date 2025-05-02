import os
import logging
import pandas as pd
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
logger = logging.getLogger("tft_trainer")

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
    """Load and prepare training data from the database"""
    try:
        conn = get_db_connection()
        
        # Query to get all features
        query = """
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
        """
        
        # Read data into DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Define feature columns (excluding timestamp and target-related columns)
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
        
        # Ensure all features are numeric
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df.dropna(inplace=True)
        
        # Normalize features
        for col in feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std if std > 0 else 1)
        
        # Calculate target (price change)
        df['target'] = df['next_price_1min'] / df['mid_price'] - 1
        
        logger.info(f"Processed {len(df)} samples with features: {feature_columns}")
        return df, feature_columns
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def train_model(
    train_loader,
    val_loader,
    test_loader,
    num_features,
    max_epochs=50,
    gpus=0
):
    """Train the TFT model"""
    
    # Initialize model with correct input dimensions
    model = TemporalFusionTransformer(
        time_varying_features=10,  # Number of features in our dataset
        hidden_size=64,
        num_heads=4,
        dropout=0.1,
        learning_rate=1e-3,
        context_length=10,  # Using 10 minutes of history
        prediction_length=1  # Predicting 1 minute ahead
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=True
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    trainer.test(model, dataloaders=test_loader)
    
    return model, trainer

def main():
    logger.info("Starting TFT model training")
    
    # Load data
    df, feature_columns = load_training_data()
    if df is None or feature_columns is None:
        logger.error("Failed to load training data")
        return
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_columns)} features")
    
    # Create data loaders with shorter context length
    train_loader, val_loader, test_loader = create_data_loaders(
        data=df,
        context_length=10,  # Reduced from 60 to 10 minutes of history
        prediction_length=1,  # 1-minute prediction
        batch_size=8,  # Reduced batch size due to smaller dataset
        train_ratio=0.7,
        val_ratio=0.15,
        target_column='target'
    )
    
    # Train model with adjusted parameters
    model, trainer = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=len(feature_columns),
        max_epochs=50,  # Reduced epochs due to smaller dataset
        gpus=0  # Use CPU
    )
    
    logger.info("Model training completed")
    
    # Save model
    model_path = f"models/tft_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 