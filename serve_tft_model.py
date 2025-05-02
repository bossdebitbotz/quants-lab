import os
import logging
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from models.improved_tft import TemporalFusionTransformer
from train_improved_tft import get_db_connection, prepare_data_for_tft
from evaluate_improved_tft import load_model
import psycopg2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_model_server")

# Initialize Flask app
app = Flask(__name__)

# Global variables to store model and related metadata
MODEL = None
MODEL_PATH = None
FEATURE_DEFINITIONS = None
CONTEXT_LENGTH = None
PREDICTION_LENGTH = None
SCALERS = None
LAST_DATA_REFRESH = None
DATA_REFRESH_INTERVAL = timedelta(minutes=5)  # Refresh data every 5 minutes

def load_latest_model():
    """Load the latest model from the models directory"""
    global MODEL, MODEL_PATH, FEATURE_DEFINITIONS, CONTEXT_LENGTH, PREDICTION_LENGTH
    
    # Find the latest model file
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('improved_tft_') and f.endswith('.pt')]
    
    if not model_files:
        logger.error("No improved TFT models found in models directory")
        return False
    
    # Sort by timestamp in filename (assuming format improved_tft_YYYYMMDD_HHMMSS.pt)
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    model_path = os.path.join(models_dir, latest_model)
    
    # Only reload if it's a different model
    if model_path != MODEL_PATH:
        logger.info(f"Loading model from {model_path}")
        try:
            model, metadata, feature_definitions = load_model(model_path)
            
            if model is None:
                logger.error("Failed to load model")
                return False
            
            MODEL = model
            MODEL_PATH = model_path
            FEATURE_DEFINITIONS = feature_definitions
            CONTEXT_LENGTH = model.hparams.context_length
            PREDICTION_LENGTH = model.hparams.prediction_length
            
            logger.info(f"Successfully loaded model: {latest_model}")
            logger.info(f"Context length: {CONTEXT_LENGTH}, Prediction length: {PREDICTION_LENGTH}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    return True  # Model already loaded

def get_latest_data():
    """Get the latest data from the database for predictions"""
    global LAST_DATA_REFRESH, SCALERS
    
    # Check if we need to refresh data
    current_time = datetime.now()
    if LAST_DATA_REFRESH is None or (current_time - LAST_DATA_REFRESH) > DATA_REFRESH_INTERVAL:
        try:
            conn = get_db_connection()
            if conn is None:
                logger.error("Failed to connect to database")
                return None
            
            # Get context_length + 1 records to have enough context for prediction
            # Adding a buffer to ensure we have enough data
            buffer = 10  # Additional records to fetch as buffer
            records_to_fetch = CONTEXT_LENGTH + buffer
            
            # Query to get the latest data
            query = """
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                bid_vol,
                ask_vol,
                imbalance,
                spread_mean,
                spread_pct_mean,
                new_bid_orders,
                new_ask_orders,
                canceled_bid_orders,
                canceled_ask_orders,
                executed_bid_orders,
                executed_ask_orders,
                buy_sell_imbalance,
                returns_10sec,
                returns_30sec,
                returns_1min,
                volatility_1min,
                target_10sec as target
            FROM tft_features_10sec
            ORDER BY timestamp DESC
            LIMIT %s
            """
            
            # Read data into DataFrame
            df = pd.read_sql_query(query, conn, params=[records_to_fetch])
            
            # Sort in ascending order of timestamp to maintain time series order
            df = df.sort_values('timestamp')
            
            # Handle missing values if any
            df = df.fillna(0)
            
            # Add derived features (similar to training process)
            # Extract hour information as categorical features
            if isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                
                # Convert to 5-minute buckets to reduce cardinality
                df['minute_bucket'] = (df['minute'] // 5).astype(int)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Prepare data for TFT (normalize using the same process as training)
            df_processed, _, scalers = prepare_data_for_tft(
                df,
                context_length=CONTEXT_LENGTH,
                prediction_length=PREDICTION_LENGTH,
                scalers=SCALERS
            )
            
            # Update scalers if this is the first time
            if SCALERS is None:
                SCALERS = scalers
            
            LAST_DATA_REFRESH = current_time
            
            logger.info(f"Fetched {len(df)} records of latest data")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            if conn:
                conn.close()
    else:
        logger.debug("Using cached data (refresh not needed yet)")
        return None  # Indicating no refresh needed

def make_prediction(df):
    """Make a prediction using the loaded model and latest data"""
    from models.improved_tft import ImprovedMarketDataset
    from torch.utils.data import DataLoader
    
    try:
        if df is None or len(df) < CONTEXT_LENGTH:
            logger.error(f"Not enough data for prediction. Need at least {CONTEXT_LENGTH} records.")
            return None
        
        # Use only the most recent context_length records
        if len(df) > CONTEXT_LENGTH:
            df = df.iloc[-CONTEXT_LENGTH:]
        
        # Create dataset with a single sample
        eval_dataset = ImprovedMarketDataset(
            data=df,
            context_length=CONTEXT_LENGTH,
            prediction_length=PREDICTION_LENGTH,
            static_categorical_cols=FEATURE_DEFINITIONS['static_categorical_cols'],
            static_real_cols=FEATURE_DEFINITIONS['static_real_cols'],
            temporal_categorical_cols=FEATURE_DEFINITIONS['temporal_categorical_cols'],
            temporal_real_cols=FEATURE_DEFINITIONS['temporal_real_cols'],
            target_col=FEATURE_DEFINITIONS['target_col']
        )
        
        # Create dataloader with batch size 1
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # Set model to evaluation mode
        MODEL.eval()
        
        # Make prediction
        with torch.no_grad():
            for batch in eval_loader:
                # Extract data
                static_categorical_features = batch.get('static_categorical_features', None)
                static_real_features = batch.get('static_real_features', None)
                temporal_categorical_features = batch.get('temporal_categorical_features', None)
                temporal_real_features = batch.get('temporal_real_features', None)
                
                # Forward pass
                output = MODEL(
                    static_categorical_features=static_categorical_features,
                    static_real_features=static_real_features,
                    temporal_categorical_features=temporal_categorical_features,
                    temporal_real_features=temporal_real_features
                )
                
                # Extract prediction
                if isinstance(output, dict) and 'prediction' in output:
                    # If model returns a dictionary with 'prediction' key
                    prediction = output['prediction'].detach().cpu().numpy()
                    
                    # Get quantile predictions if available
                    quantile_predictions = None
                    if hasattr(MODEL, 'quantiles') and 'quantile_predictions' in output:
                        quantile_predictions = {}
                        for i, q in enumerate(MODEL.quantiles):
                            quantile_predictions[str(q)] = float(output['quantile_predictions'][0, 0, i].item())
                    
                    # Return main prediction and quantiles if available
                    result = {
                        'prediction': float(prediction[0, 0]),
                        'timestamp': datetime.now().isoformat(),
                    }
                    
                    if quantile_predictions:
                        result['quantiles'] = quantile_predictions
                    
                    return result
                elif isinstance(output, torch.Tensor):
                    # Direct tensor output
                    prediction = output.detach().cpu().numpy()
                    return {
                        'prediction': float(prediction[0, 0]),
                        'timestamp': datetime.now().isoformat()
                    }
                
        logger.error("Failed to get prediction from model")
        return None
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the API"""
    if MODEL is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'ok',
        'model': os.path.basename(MODEL_PATH) if MODEL_PATH else 'none',
        'last_data_refresh': LAST_DATA_REFRESH.isoformat() if LAST_DATA_REFRESH else None
    })

@app.route('/predict', methods=['GET'])
def predict():
    """Endpoint to get a prediction"""
    # Make sure model is loaded
    if MODEL is None:
        success = load_latest_model()
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load model'
            }), 500
    
    # Get latest data
    df = get_latest_data()
    
    # Make prediction
    prediction = make_prediction(df)
    
    if prediction is None:
        return jsonify({
            'status': 'error',
            'message': 'Failed to make prediction'
        }), 500
    
    # Return prediction
    return jsonify({
        'status': 'ok',
        'data': prediction
    })

@app.route('/refresh', methods=['POST'])
def refresh_model():
    """Endpoint to force a refresh of the model and data"""
    global LAST_DATA_REFRESH
    
    # Force model reload
    success = load_latest_model()
    
    # Force data refresh
    LAST_DATA_REFRESH = None
    df = get_latest_data()
    
    if not success or df is None:
        return jsonify({
            'status': 'error',
            'message': 'Failed to refresh model or data'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'message': 'Model and data refreshed successfully',
        'model': os.path.basename(MODEL_PATH) if MODEL_PATH else 'none'
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Endpoint to get the current model configuration"""
    if MODEL is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    # Return model configuration
    config = {
        'model_path': MODEL_PATH,
        'context_length': CONTEXT_LENGTH,
        'prediction_length': PREDICTION_LENGTH,
        'feature_definitions': FEATURE_DEFINITIONS,
        'last_data_refresh': LAST_DATA_REFRESH.isoformat() if LAST_DATA_REFRESH else None
    }
    
    return jsonify({
        'status': 'ok',
        'config': config
    })

def init_app():
    """Initialize the Flask application"""
    # Load model on startup
    load_latest_model()
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Start TFT model serving API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--model', type=str, help='Specific model file to load')
    
    args = parser.parse_args()
    
    # If specific model is specified, load it
    if args.model:
        MODEL_PATH = args.model
        load_latest_model()
    else:
        # Otherwise load the latest model
        load_latest_model()
    
    # Start the server
    app = init_app()
    app.run(host=args.host, port=args.port, debug=args.debug) 