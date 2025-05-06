import os
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables or .env file"""
    # Try to load from .env file if it exists
    load_dotenv()
    
    logger.info("Loading TFT bot configuration...")
    
    config = {
        # Model Configuration
        'DIR_MODEL_PATH': os.getenv('DIR_MODEL_PATH', '/app/models/directional_tft_20250502_134913.pt'),
        'DOWN_MODEL_PATH': os.getenv('DOWN_MODEL_PATH', '/app/models/downward_specialist_tft_20250502_140300.pt'),
        'CONTEXT_LENGTH': int(os.getenv('CONTEXT_LENGTH', 20)),
        'EXPECTED_FEATURE_COUNT': int(os.getenv('EXPECTED_FEATURE_COUNT', 64)),
        
        # Ensemble & Trading Logic
        'ENSEMBLE_METHOD': os.getenv('ENSEMBLE_METHOD', 'selective'),
        'ENSEMBLE_THRESHOLD': float(os.getenv('ENSEMBLE_THRESHOLD', 0.002)),
        'TRADING_THRESHOLD': float(os.getenv('TRADING_THRESHOLD', 0.003)),
        'POSITION_SIZING_ENABLED': os.getenv('POSITION_SIZING_ENABLED', 'true').lower() == 'true',
        'POSITION_SIZE_BASE': float(os.getenv('POSITION_SIZE_BASE', 0.1)),
        'POSITION_SIZE_FACTOR': float(os.getenv('POSITION_SIZE_FACTOR', 0.4)),
        'POSITION_SIZE_FIXED': float(os.getenv('POSITION_SIZE_FIXED', 0.3)),
        'STOP_LOSS': float(os.getenv('STOP_LOSS', 0.002)),
        'TAKE_PROFIT': float(os.getenv('TAKE_PROFIT', 0.004)),
        'MAX_HOLDING_PERIOD': int(os.getenv('MAX_HOLDING_PERIOD', 8)),
        'USE_DYNAMIC_THRESHOLD': os.getenv('USE_DYNAMIC_THRESHOLD', 'true').lower() == 'true',
        'FEE_PER_TRADE': float(os.getenv('FEE_PER_TRADE', 0.0001)),
        
        # Data Source
        'TRADING_PAIR': os.getenv('TRADING_PAIR', 'WLD-USDT'),
        'EXCHANGE_SYMBOL': os.getenv('EXCHANGE_SYMBOL', 'WLDUSDT'),
        'PREDICTION_INTERVAL_SECONDS': int(os.getenv('PREDICTION_INTERVAL_SECONDS', 10)),
        'DEPTH_STREAM_INTERVAL': os.getenv('DEPTH_STREAM_INTERVAL', '100ms'),
        
        # Binance API
        'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY', ''),
        'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET', ''),
        'BINANCE_BASE_URL': os.getenv('BINANCE_BASE_URL', 'https://fapi.binance.com'),
        
        # Database (TimescaleDB)
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': int(os.getenv('DB_PORT', 5441)),
        'DB_NAME': os.getenv('DB_NAME', 'backtest_db'),
        'DB_USER': os.getenv('DB_USER', 'backtest_user'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'backtest_password'),
        
        # Execution
        'ENABLE_LIVE_TRADING': os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true',
    }
    
    # Validation
    if config['ENABLE_LIVE_TRADING'] and (not config['BINANCE_API_KEY'] or not config['BINANCE_API_SECRET']):
        logger.warning("Live trading enabled but API credentials missing! Disabling live trading.")
        config['ENABLE_LIVE_TRADING'] = False
    
    # Log config (hiding sensitive info)
    masked_config = config.copy()
    if masked_config['BINANCE_API_KEY']:
        masked_config['BINANCE_API_KEY'] = '***' + masked_config['BINANCE_API_KEY'][-4:]
    if masked_config['BINANCE_API_SECRET']:
        masked_config['BINANCE_API_SECRET'] = '***' + masked_config['BINANCE_API_SECRET'][-4:]
    masked_config['DB_PASSWORD'] = '********' if masked_config['DB_PASSWORD'] else None
    
    logger.info(f"Configuration loaded: {json.dumps(masked_config, indent=2)}")
    return config 