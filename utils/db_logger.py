import logging
import json
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from datetime import datetime

logger = logging.getLogger(__name__)

class DBLogger:
    """Database logger for TFT Bot to interact with TimescaleDB"""
    
    def __init__(self, config):
        self.config = config
        self.pool = None
        
    async def initialize(self):
        """Initialize the database connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config['DB_HOST'],
                port=self.config['DB_PORT'],
                database=self.config['DB_NAME'],
                user=self.config['DB_USER'],
                password=self.config['DB_PASSWORD']
            )
            
            # Test connection
            conn = self._get_connection()
            if conn:
                logger.info(f"Successfully connected to database {self.config['DB_NAME']} on {self.config['DB_HOST']}:{self.config['DB_PORT']}")
                self._return_connection(conn)
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connections closed")
    
    def _get_connection(self):
        """Get a connection from the pool"""
        if not self.pool:
            logger.error("Database pool not initialized")
            return None
            
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        if self.pool:
            self.pool.putconn(conn)
    
    async def log_prediction(self, timestamp, trading_pair, dir_pred, down_pred, ensemble_pred, feature_context=None):
        """Log a prediction to the database
        
        Args:
            timestamp: Prediction timestamp
            trading_pair: The trading pair (e.g., 'WLD-USDT')
            dir_pred: Directional model prediction
            down_pred: Downward specialist prediction
            ensemble_pred: Ensemble prediction
            feature_context: Optional dict of feature values used
            
        Returns:
            The prediction_id if successful, None otherwise
        """
        # Validate input data
        if not timestamp:
            logger.error("Missing timestamp for prediction log")
            return None
            
        if not trading_pair:
            logger.error("Missing trading_pair for prediction log")
            return None
            
        # Ensure predictions are floats or can be converted to floats
        try:
            dir_pred = float(dir_pred) if dir_pred is not None else 0.0
            down_pred = float(down_pred) if down_pred is not None else 0.0
            ensemble_pred = float(ensemble_pred) if ensemble_pred is not None else 0.0
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid prediction values (must be numeric): {e}")
            return None
            
        # Validate feature_context is a dict if provided
        if feature_context is not None and not isinstance(feature_context, dict):
            logger.error(f"feature_context must be a dict, got {type(feature_context)}")
            return None
        
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for logging prediction")
            return None
            
        try:
            with conn.cursor() as cursor:
                # Prepare feature context as JSON
                feature_json = None
                if feature_context:
                    try:
                        feature_json = json.dumps(feature_context)
                    except Exception as e:
                        logger.error(f"Failed to serialize feature_context to JSON: {e}")
                        # Continue without feature context
                
                # Insert the prediction
                cursor.execute("""
                    INSERT INTO tft_predictions
                    (prediction_timestamp, trading_pair, dir_prediction, 
                     down_prediction, ensemble_prediction, feature_context)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    trading_pair,
                    dir_pred,
                    down_pred,
                    ensemble_pred,
                    feature_json
                ))
                
                prediction_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"Logged prediction {prediction_id} for {trading_pair}: dir={dir_pred:.6f}, down={down_pred:.6f}, ensemble={ensemble_pred:.6f}")
                return prediction_id
                
        except psycopg2.errors.InvalidDatetimeFormat as e:
            logger.error(f"Invalid timestamp format: {e}")
            if conn:
                conn.rollback()
            return None
        except psycopg2.errors.InvalidTextRepresentation as e:
            logger.error(f"Invalid data type in prediction log: {e}")
            if conn:
                conn.rollback()
            return None
        except Exception as e:
            logger.error(f"Error logging prediction: {type(e).__name__} - {str(e)}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def log_trade_decision(self, timestamp, trading_pair, prediction_id, decision, 
                                target_price, prediction_value, reason=None):
        """Log a trade decision to the database
        
        Args:
            timestamp: Decision timestamp
            trading_pair: The trading pair
            prediction_id: ID of the related prediction
            decision: Decision string (e.g., 'ENTER_LONG', 'EXIT_SHORT', 'HOLD')
            target_price: Current price at decision time
            prediction_value: The ensemble prediction value
            reason: Optional reason for the decision
            
        Returns:
            The decision_id if successful, None otherwise
        """
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trade_decisions
                    (decision_timestamp, trading_pair, prediction_id, 
                     decision, target_price, prediction_value, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    trading_pair,
                    prediction_id,
                    decision,
                    target_price,
                    prediction_value,
                    reason
                ))
                
                decision_id = cursor.fetchone()[0]
                conn.commit()
                logger.debug(f"Logged trade decision {decision_id}: {decision} for {trading_pair}")
                return decision_id
                
        except Exception as e:
            logger.error(f"Error logging trade decision: {str(e)}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def log_executed_trade(self, decision_id, order_data):
        """Log an executed trade to the database
        
        Args:
            decision_id: ID of the related trade decision
            order_data: Dict with order execution details
            
        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        if not conn:
            return False
            
        try:
            with conn.cursor() as cursor:
                # Use SQL function to convert timestamp directly in database
                cursor.execute("""
                    INSERT INTO executed_trades
                    (decision_id, exchange_order_id, trading_pair, side, 
                     order_type, status, requested_quantity, filled_quantity,
                     average_fill_price, commission, commission_asset, 
                     transaction_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                     datetime_from_ms(%s))
                """, (
                    decision_id,
                    order_data.get('orderId'),
                    order_data.get('symbol'),
                    order_data.get('side'),
                    order_data.get('type'),
                    order_data.get('status'),
                    order_data.get('origQty'),
                    order_data.get('executedQty'),
                    order_data.get('avgPrice'),
                    order_data.get('commission'),
                    order_data.get('commissionAsset'),
                    order_data.get('updateTime') or order_data.get('time')
                ))
                
                conn.commit()
                logger.info(f"Logged executed trade for order ID: {order_data.get('orderId')}")
                return True
                
        except Exception as e:
            logger.error(f"Error logging executed trade: {str(e)}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)