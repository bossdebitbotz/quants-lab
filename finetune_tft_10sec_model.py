import os
import logging
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
import psycopg2
import torch
import glob
from models.temporal_fusion_transformer import TemporalFusionTransformer, create_data_loaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_10sec_finetuner")

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
        SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM tft_features_10sec
        """
        with conn.cursor() as cursor:
            cursor.execute(date_query)
            min_date, max_date, count = cursor.fetchone()
        
        if count is None or count == 0:
            logger.error("No data found in tft_features_10sec table")
            return None, None
        
        logger.info(f"Found {count} records from {min_date} to {max_date}")
        
        # Query to get all features from 10-second bars
        query = """
        SELECT 
            timestamp,
            open, high, low, close, 
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
        ORDER BY timestamp
        """
        
        # Read data into DataFrame
        df = pd.read_sql_query(query, conn)
        
        logger.info(f"Retrieved {len(df)} rows of raw data")
        
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
        
        # Ensure all features are numeric and fill NaN values with 0
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna(0)
        
        # Normalize features and handle zeros
        df_scaled = df.copy()
        for col in feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            # Avoid division by zero
            if std == 0 or np.isnan(std):
                logger.warning(f"Standard deviation for {col} is {std}, setting feature to zero")
                df_scaled[col] = 0
            else:
                df_scaled[col] = (df[col] - mean) / std
        
        # Replace invalid values with 0
        df_scaled = df_scaled.replace([np.inf, -np.inf, np.nan], 0)
        
        # Rename target column for consistency
        df_scaled['target'] = df_scaled['target_10sec']
        
        logger.info(f"Processed {len(df_scaled)} samples with {len(feature_columns)} features")
        
        return df_scaled, feature_columns[:21]  # Use only the first 21 features
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    finally:
        if conn:
            conn.close()

def find_latest_model():
    """Find the latest trained 10-second model"""
    # Check for saved model files
    model_files = glob.glob("models/tft_10sec_model_*.pt")
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_files[0]
    
    # Check for checkpoint files
    checkpoint_files = glob.glob("checkpoints/10sec/tft-10sec-*.ckpt")
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoint_files[0]
    
    return None

def finetune_model(
    train_loader,
    val_loader, 
    test_loader,
    num_features,
    model_path=None,
    max_epochs=30
):
    """Finetune the TFT model with bias correction"""
    logger.info(f"Fine-tuning model with {num_features} features")
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        
        # First initialize a simple model to check the architecture
        temp_model = None
        if model_path.endswith('.pt'):
            state_dict = torch.load(model_path)
            # Check if the model has the new or old architecture based on state dict
            has_new_arch = 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict
            
            logger.info(f"Detected {'new' if has_new_arch else 'old'} model architecture")
            
            # Initialize with matching architecture
            if has_new_arch:
                # New Transformer architecture
                model = TemporalFusionTransformer(
                    time_varying_features=num_features,
                    hidden_size=128,
                    num_heads=4,
                    dropout=0.2,
                    learning_rate=1e-4,  # Lower learning rate for fine-tuning
                    context_length=30,
                    prediction_length=1,
                    bias_correction=True  # Enable bias correction
                )
            else:
                # Original architecture with compatible layer sizes
                from models.temporal_fusion_transformer import TemporalFusionTransformer as OldTFT
                model = OldTFT(
                    time_varying_features=num_features,
                    hidden_size=128,
                    dropout=0.2,
                    learning_rate=1e-4,
                    context_length=30,
                    prediction_length=1
                )
                # Add bias correction manually
                model.bias_correction = True
                model.prediction_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            
            # Add bias parameter if not present
            if 'prediction_bias' not in state_dict:
                state_dict['prediction_bias'] = torch.zeros(1)
                
            # Adjust model based on state dict
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model state dict")
            except Exception as e:
                logger.error(f"Error loading state dict: {e}")
                logger.info("Creating a simple bias correction wrapper for the existing model")
                
                # Create a simple wrapper class to apply bias correction
                class BiasCorrectingWrapper(torch.nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model
                        self.prediction_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
                        self.bias_correction = True
                        
                    def forward(self, static_features, temporal_features):
                        # Forward pass through base model
                        predictions = self.base_model(static_features, temporal_features)
                        # Apply bias correction
                        if self.bias_correction:
                            predictions = predictions - self.prediction_bias
                        return predictions
                
                # Create original model structure
                base_model = OldTFT(
                    time_varying_features=num_features,
                    hidden_size=128,
                    dropout=0.2,
                    learning_rate=1e-4,
                    context_length=30,
                    prediction_length=1
                )
                base_model.load_state_dict(state_dict, strict=True)
                
                # Wrap with bias correction
                model = BiasCorrectingWrapper(base_model)
            
        else:
            # Load from checkpoint
            try:
                model = TemporalFusionTransformer.load_from_checkpoint(model_path)
                # Make sure bias correction is enabled
                model.bias_correction = True
                if not hasattr(model, 'prediction_bias'):
                    model.prediction_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            except Exception as e:
                logger.error(f"Error loading from checkpoint: {e}")
                return None, None
    else:
        logger.warning("No existing model found, training from scratch")
        model = TemporalFusionTransformer(
            time_varying_features=num_features,
            hidden_size=128,
            num_heads=4,
            dropout=0.2,
            learning_rate=1e-4,
            context_length=30,
            prediction_length=1,
            bias_correction=True
        )
    
    # Create a simpler model just for fine-tuning the bias
    class BiasOnlyModel(pl.LightningModule):
        def __init__(self, base_model, learning_rate=1e-3):
            super().__init__()
            self.base_model = base_model
            self.prediction_bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            self.learning_rate = learning_rate
            
            # Freeze the base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False
            
        def forward(self, static_features, temporal_features):
            with torch.no_grad():
                predictions = self.base_model(static_features, temporal_features)
            # Apply bias correction
            return predictions - self.prediction_bias
        
        def training_step(self, batch, batch_idx):
            static_features = batch.get('static_features', None)
            temporal_features = batch['temporal_features']
            targets = batch['targets'].squeeze(-1)
            
            # Make predictions
            predictions = self(static_features, temporal_features).squeeze(-1)
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predictions, targets)
            self.log('train_loss', loss)
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            static_features = batch.get('static_features', None)
            temporal_features = batch['temporal_features']
            targets = batch['targets'].squeeze(-1)
            
            # Make predictions
            predictions = self(static_features, temporal_features).squeeze(-1)
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predictions, targets)
            self.log('val_loss', loss)
            
            return loss
        
        def test_step(self, batch, batch_idx):
            static_features = batch.get('static_features', None)
            temporal_features = batch['temporal_features']
            targets = batch['targets'].squeeze(-1)
            
            # Make predictions
            predictions = self(static_features, temporal_features).squeeze(-1)
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(predictions, targets)
            self.log('test_loss', loss)
            
            # Calculate MAE
            error = torch.abs(predictions - targets).mean()
            self.log('test_mae', error)
            
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam([self.prediction_bias], lr=self.learning_rate)
    
    # Check initial bias
    initial_bias = check_prediction_bias(model, test_loader)
    logger.info(f"Initial prediction bias: {initial_bias:.6f}")
    
    # Create bias-only model
    bias_only_model = BiasOnlyModel(model, learning_rate=1e-2)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/10sec',
        filename='tft-10sec-finetuned-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # Shorter patience for fine-tuning
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
    
    # Fine-tune model
    try:
        trainer.fit(
            bias_only_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Test model
        trainer.test(bias_only_model, dataloaders=test_loader)
        
        # Update the original model with the learned bias
        model.prediction_bias.data = bias_only_model.prediction_bias.data
        
        # Check final bias
        final_bias = check_prediction_bias(model, test_loader)
        logger.info(f"Final prediction bias: {final_bias:.6f}")
        logger.info(f"Learned bias correction value: {model.prediction_bias.item():.6f}")
        
        # Save model
        model_path = f"models/tft_10sec_model_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Fine-tuned model saved to {model_path}")
        
        return model, trainer
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

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
    logger.info("Starting TFT model fine-tuning on 10-second bars")
    
    # Find latest model
    model_path = find_latest_model()
    if model_path:
        logger.info(f"Found existing model: {model_path}")
    else:
        logger.warning("No existing model found")
    
    # Load data
    df, feature_columns = load_training_data()
    if df is None or feature_columns is None:
        logger.error("Failed to load training data")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        df,
        context_length=30,
        prediction_length=1,
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15,
        target_column='target'
    )
    
    # Fine-tune model
    model, trainer = finetune_model(
        train_loader,
        val_loader,
        test_loader,
        len(feature_columns),
        model_path=model_path,
        max_epochs=30
    )
    
    if model:
        logger.info("Model fine-tuning completed successfully")
    else:
        logger.error("Model fine-tuning failed")

if __name__ == "__main__":
    main() 