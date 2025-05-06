#!/usr/bin/env python3
"""
Real-time TFT Bot for Crypto Trading

This bot combines a Temporal Fusion Transformer model ensemble
with real-time order book tracking to make trading decisions.
"""

import os
import sys
import asyncio
import signal
import logging
from datetime import datetime, timedelta

# Import local modules
from config import load_config
from utils.db_logger import DBLogger
from utils.data_fetcher import DataFetcher
from utils.feature_calculator import FeatureCalculator
from utils.tft_predictor import TFTPredictor
from utils.order_book_processor import OrderBookProcessor
from utils.decision_engine import DecisionEngine
from utils.execution_handler import ExecutionHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tft_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RealTimeTFTBot")

# Global flag for controlling continuous execution
running = True

def handle_exit_signal(sig, frame):
    """Handle exit signals gracefully"""
    global running
    logger.info("Shutdown signal received. Stopping bot...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

async def startup():
    """Initialize all required components"""
    # Load configuration
    config = load_config()
    
    # Initialize database logger
    db_logger = DBLogger(config)
    if not await db_logger.initialize():
        logger.error("Failed to initialize database connection. Exiting.")
        return None
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(config)
    if not await data_fetcher.initialize():
        logger.error("Failed to initialize data fetcher. Exiting.")
        await db_logger.close()
        return None
    
    # Initialize feature calculator
    feature_calculator = FeatureCalculator(config)
    feature_calculator.initialize()
    
    # Initialize TFT predictor
    tft_predictor = TFTPredictor(config)
    if not await tft_predictor.initialize():
        logger.error("Failed to initialize TFT predictor. Exiting.")
        await data_fetcher.close()
        await db_logger.close()
        return None
    
    # Initialize order book processor
    order_book_processor = OrderBookProcessor(config, data_fetcher, db_logger)
    if not await order_book_processor.initialize():
        logger.error("Failed to initialize order book processor. Exiting.")
        await data_fetcher.close()
        await db_logger.close()
        return None
    
    # Initialize decision engine
    decision_engine = DecisionEngine(config, db_logger)
    if not await decision_engine.initialize():
        logger.error("Failed to initialize decision engine. Exiting.")
        await data_fetcher.close()
        await db_logger.close()
        return None
    
    # Initialize execution handler
    execution_handler = ExecutionHandler(config, db_logger)
    if not await execution_handler.initialize():
        logger.error("Failed to initialize execution handler. Exiting.")
        await data_fetcher.close()
        await db_logger.close()
        return None
    
    logger.info("Bot initialization complete")
    return {
        'config': config,
        'db_logger': db_logger,
        'data_fetcher': data_fetcher,
        'feature_calculator': feature_calculator,
        'tft_predictor': tft_predictor,
        'order_book_processor': order_book_processor,
        'decision_engine': decision_engine,
        'execution_handler': execution_handler
    }

async def prediction_loop(components):
    """Main prediction and trading loop"""
    config = components['config']
    db_logger = components['db_logger']
    data_fetcher = components['data_fetcher']
    feature_calculator = components['feature_calculator']
    tft_predictor = components['tft_predictor']
    decision_engine = components['decision_engine']
    execution_handler = components['execution_handler']
    
    # Define trading pair and interval
    trading_pair = config['TRADING_PAIR']
    interval_seconds = config['PREDICTION_INTERVAL_SECONDS']
    
    # Last decision time
    last_decision_time = None
    
    logger.info(f"Starting prediction loop for {trading_pair} at {interval_seconds}s intervals")
    
    while running:
        try:
            logger.info(f"Starting prediction cycle at {datetime.now()}")
            
            # Step 1: Fetch latest OHLCV data
            ohlcv_df = await data_fetcher.get_latest_ohlcv(
                trading_pair, 
                interval_seconds, 
                limit=max(50, config['CONTEXT_LENGTH'] * 2)  # Get enough historical data
            )
            
            if ohlcv_df is None or len(ohlcv_df) < config['CONTEXT_LENGTH']:
                logger.warning(f"Not enough OHLCV data for prediction (need {config['CONTEXT_LENGTH']} rows)")
                await asyncio.sleep(interval_seconds)
                continue
            
            # Step 2: Update features with new data
            if not feature_calculator.update_with_ohlcv(ohlcv_df):
                logger.warning("Failed to update features with OHLCV data")
                await asyncio.sleep(interval_seconds)
                continue
            
            # Step 3: Add order book features
            current_order_book = data_fetcher.get_current_order_book()
            feature_calculator.add_order_book_features(current_order_book)
            
            # Step 4: Prepare model input
            model_input = feature_calculator.prepare_model_input()
            if not model_input:
                logger.warning("Failed to prepare model input")
                await asyncio.sleep(interval_seconds)
                continue
            
            # Step 5: Run TFT prediction
            prediction = tft_predictor.predict(model_input)
            if not prediction:
                logger.warning("Failed to generate prediction")
                await asyncio.sleep(interval_seconds)
                continue
            
            # Log prediction details
            logger.info(f"Generated prediction - Dir: {prediction['dir_prediction']:.6f}, Down: {prediction['down_prediction']:.6f}, Ensemble: {prediction['ensemble_prediction']:.6f}")
            
            # Step 6: Log prediction to database
            await db_logger.log_prediction(
                timestamp=prediction['timestamp'],
                trading_pair=trading_pair,
                dir_pred=prediction['dir_prediction'],
                down_pred=prediction['down_prediction'],
                ensemble_pred=prediction['ensemble_prediction'],
                feature_context=None  # Could store feature values here if desired
            )
            
            # Get current price
            current_price = prediction['last_price']
            logger.info(f"Current price: {current_price}")
            
            # Step 7: Make trading decision
            decision = await decision_engine.make_decision(prediction, current_price)
            if not decision:
                logger.warning("Failed to make trading decision")
                await asyncio.sleep(interval_seconds)
                continue
            
            logger.info(f"Decision: {decision['decision']}, Reason: {decision.get('reason')}")
            
            # Step 8: Execute trading decision
            if decision['decision'] != 'HOLD':
                logger.info(f"Executing decision: {decision['decision']}")
                execution_result = await execution_handler.execute_decision(decision)
                if execution_result:
                    logger.info(f"Executed {decision['decision']} with result: {execution_result.get('status')}")
                    # Reload position state from database after trade execution
                    logger.info("Reloading position state after trade execution")
                    await decision_engine.reload_position_state()
            
            # Step 9: Increment holding period if in a position
            await decision_engine.increment_holding_period()
            
            # Step 10: Wait until next interval
            now = datetime.now()
            last_decision_time = now
            
            # Calculate time to sleep - ensure we don't go negative if processing took long
            time_elapsed = (datetime.now() - now).total_seconds()
            sleep_time = max(0, interval_seconds - time_elapsed)
            
            logger.info(f"Prediction cycle completed in {time_elapsed:.2f}s. Sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in prediction loop: {str(e)}", exc_info=True)
            await asyncio.sleep(interval_seconds)

async def main():
    """Main entry point"""
    try:
        # Initialize components
        components = await startup()
        if not components:
            return 1
        
        config = components['config']
        db_logger = components['db_logger']
        data_fetcher = components['data_fetcher']
        order_book_processor = components['order_book_processor']
        
        logger.info(f"Starting TFT Bot for {config['TRADING_PAIR']}")
        logger.info(f"Database connection successful: {config['DB_HOST']}:{config['DB_PORT']}")
        
        # Start order book processor in a separate task
        orderbook_task = asyncio.create_task(
            order_book_processor.start_processing()
        )
        
        # Start prediction loop in main task
        prediction_task = asyncio.create_task(
            prediction_loop(components)
        )
        
        # Wait for tasks to complete or for interruption
        pending = {orderbook_task, prediction_task}
        while running and pending:
            done, pending = await asyncio.wait(
                pending, 
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                if task.exception():
                    logger.error(f"Task failed with exception: {task.exception()}")
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Critical error in main function: {e}", exc_info=True)
        return 1
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        
        if 'order_book_processor' in locals() and order_book_processor:
            await order_book_processor.stop_processing()
            
        if 'execution_handler' in locals() and 'components' in locals() and 'execution_handler' in components:
            await components['execution_handler'].close()
            
        if 'data_fetcher' in locals() and 'components' in locals() and 'data_fetcher' in components:
            await components['data_fetcher'].close()
            
        if 'db_logger' in locals() and 'components' in locals() and 'db_logger' in components:
            await components['db_logger'].close()
        
        logger.info("Bot stopped.")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 