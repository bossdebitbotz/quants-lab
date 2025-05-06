#!/usr/bin/env python3
"""
Restart script for the TFT bot with the new rate limiting and order book processing improvements
"""
import os
import sys
import time
import logging
import torch
import subprocess
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_restart.log")
    ]
)
logger = logging.getLogger("bot_restarter")

def main():
    """Main function to restart the TFT bot with new rate limiting configurations"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Restart TFT bot with new rate limiting configurations")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode with live trading disabled")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode without connecting to exchange API")
    parser.add_argument("--websocket-only", action="store_true", help="Use only WebSocket connections to avoid CloudFront blocking")
    args = parser.parse_args()

    logger.info("Restarting TFT bot with new rate limiting and order book processing changes")
    
    # Set up model paths with absolute paths
    dir_model_path = os.path.abspath("/Volumes/Elite_Frosty /quants-lab/models/directional_tft_20250502_134913.pt")
    down_model_path = os.path.abspath("/Volumes/Elite_Frosty /quants-lab/models/downward_specialist_tft_20250502_140300.pt")
    
    # Configure environment variables for the new bot run
    os.environ['DIR_MODEL_PATH'] = dir_model_path
    os.environ['DOWN_MODEL_PATH'] = down_model_path
    os.environ['TORCH_USE_RTLD_GLOBAL'] = 'TRUE'
    
    # Apply new rate limiting and order book configs
    os.environ['DEPTH_STREAM_INTERVAL'] = '1000ms'  # Use 1000ms instead of 100ms
    os.environ['SMALL_GAP_TOLERANCE'] = '1000'
    os.environ['MEDIUM_GAP_TOLERANCE'] = '5000'
    os.environ['MIN_SNAPSHOT_INTERVAL'] = '300'
    os.environ['MEDIUM_REFRESH_INTERVAL'] = '120'
    os.environ['CIRCUIT_BREAKER_THRESHOLD'] = '5'
    os.environ['CIRCUIT_BREAKER_DELAY'] = '300'
    os.environ['USE_WEBSOCKET_FOR_KLINES'] = 'true'
    os.environ['MAX_RETRIES'] = '3'
    
    # WebSocket-specific configurations to avoid CloudFront blocking
    if args.websocket_only:
        logger.info("Running in WebSocket-only mode to avoid CloudFront blocking")
        os.environ['USE_WEBSOCKET_FOR_DEPTH'] = 'true'
        os.environ['USE_WEBSOCKET_FOR_KLINES'] = 'true'
        os.environ['USE_WEBSOCKET_FOR_TRADES'] = 'true'
        os.environ['USE_REST_FALLBACK'] = 'false'
        # Direct WebSocket URLs to avoid CloudFront
        os.environ['WEBSOCKET_BASE_URL'] = 'wss://stream.binance.com:9443/ws'
        # Set custom user agent
        os.environ['CUSTOM_USER_AGENT'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    else:
        # Default mode - use both REST and WebSocket with fallbacks
        os.environ['USE_WEBSOCKET_FOR_DEPTH'] = 'true'
        os.environ['USE_REST_FALLBACK'] = 'true'
    
    # Configure operating mode
    if args.simulation:
        # Simulation mode - uses local data and doesn't connect to exchange
        os.environ['SIMULATION_MODE'] = 'true'
        os.environ['ENABLE_LIVE_TRADING'] = 'false'
        logger.info("Running in SIMULATION MODE - using local data, no API connections")
    elif args.test_mode:
        # Test mode - connects to API but doesn't execute real trades
        os.environ['SIMULATION_MODE'] = 'false'
        os.environ['ENABLE_LIVE_TRADING'] = 'false'
        os.environ['BINANCE_BASE_URL'] = 'https://testnet.binancefuture.com'  # Use testnet for testing
        logger.info("Running in TEST MODE - connecting to API but trading disabled")
    else:
        # Live mode - normal operation with real trading
        os.environ['SIMULATION_MODE'] = 'false'
        os.environ['ENABLE_LIVE_TRADING'] = 'true'
        os.environ['BINANCE_BASE_URL'] = 'https://fapi.binance.com'
        logger.info("Running in LIVE MODE - live trading enabled")
    
    # Database configuration
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_PORT'] = '5441'
    os.environ['DB_NAME'] = 'backtest_db'
    os.environ['DB_USER'] = 'backtest_user'
    os.environ['DB_PASSWORD'] = 'backtest_password'
    
    # Loading PyTorch and adding security exceptions
    logger.info("Configuring PyTorch...")
    torch.serialization.add_safe_globals(["lightning_fabric.utilities.data.AttributeDict"])
    
    # Print the configuration to verify
    logger.info(f"Using the following configuration:")
    logger.info(f"- Directional model: {os.environ['DIR_MODEL_PATH']}")
    logger.info(f"- Downward model: {os.environ['DOWN_MODEL_PATH']}")
    logger.info(f"- Depth stream interval: {os.environ['DEPTH_STREAM_INTERVAL']}")
    logger.info(f"- Small gap tolerance: {os.environ['SMALL_GAP_TOLERANCE']}")
    logger.info(f"- Medium gap tolerance: {os.environ['MEDIUM_GAP_TOLERANCE']}")
    logger.info(f"- Min snapshot interval: {os.environ['MIN_SNAPSHOT_INTERVAL']}s")
    logger.info(f"- Circuit breaker threshold: {os.environ['CIRCUIT_BREAKER_THRESHOLD']}")
    logger.info(f"- Circuit breaker delay: {os.environ['CIRCUIT_BREAKER_DELAY']}s")
    logger.info(f"- Using WebSocket for klines: {os.environ['USE_WEBSOCKET_FOR_KLINES']}")
    
    # Display operating mode details
    if args.websocket_only:
        logger.info(f"- WebSocket-only mode: Enabled")
        logger.info(f"- WebSocket base URL: {os.environ['WEBSOCKET_BASE_URL']}")
    
    if args.simulation:
        logger.info(f"- Mode: SIMULATION (no API connections)")
    else:
        logger.info(f"- Base URL: {os.environ['BINANCE_BASE_URL']}")
        logger.info(f"- Live trading: {os.environ['ENABLE_LIVE_TRADING']}")
    
    # Add a small pause to ensure any previous processes are completely terminated
    logger.info("Waiting 5 seconds before starting the bot...")
    time.sleep(5)
    
    # Start the bot
    logger.info("Starting the TFT bot with new rate limiting configs...")
    
    # Choose the appropriate script based on simulation mode
    if args.simulation:
        bot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_tft_bot", "simulation_bot.py")
        if not os.path.exists(bot_script):
            # Fallback to the regular bot with simulation flag if simulation script doesn't exist
            bot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_tft_bot", "realtime_tft_bot.py")
    else:
        bot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_tft_bot", "realtime_tft_bot.py")
    
    # Execute the bot using execv to replace the current process
    os.execv(sys.executable, [sys.executable, bot_script])

if __name__ == "__main__":
    main() 