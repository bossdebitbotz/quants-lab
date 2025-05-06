from datetime import datetime
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
        self.conn_pool = None
        
    async def initialize(self):
        """Initialize database connection pool and verify tables"""
        try:
            # Create connection pool
            self.conn_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.config['DB_HOST'],
                port=self.config['DB_PORT'],
                database=self.config['DB_NAME'],
                user=self.config['DB_USER'],
                password=self.config['DB_PASSWORD']
            )
            
            if not self.conn_pool:
                logger.error("Failed to create database connection pool")
                return False
            
            logger.info(f"Connected to database: {self.config['DB_HOST']}:{self.config['DB_PORT']}")
            
            # Verify tables exist and create them if needed
            await self._create_tables()
            
            # Verify position state table schema
            await self._verify_position_state_schema()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing database connection: {str(e)}")
            return False
        
    async def _verify_position_state_schema(self):
        """Verify and update position_state table schema if needed"""
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for schema verification")
            return False
        
        try:
            with conn.cursor() as cursor:
                # Check for last_update_timestamp column
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns 
                    WHERE table_name = 'position_state'
                    AND column_name = 'last_update_timestamp'
                """)
                
                has_last_update = cursor.fetchone() is not None
                
                if not has_last_update:
                    logger.info("Adding last_update_timestamp column to position_state table")
                    cursor.execute("""
                        ALTER TABLE position_state
                        ADD COLUMN last_update_timestamp TIMESTAMP DEFAULT NOW()
                    """)
                    conn.commit()
                    logger.info("Schema update completed")
                else:
                    logger.info("Position state table schema is up to date")
                
                return True
        except Exception as e:
            logger.error(f"Error verifying position state schema: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    async def close(self):
        """Close all database connections"""
        try:
            if hasattr(self, 'conn_pool') and self.conn_pool:
                self.conn_pool.closeall()
                logger.info("Closed all database connections")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
    
    def _get_connection(self):
        """Get a connection from the pool"""
        try:
            return self.conn_pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection from pool: {str(e)}")
            return None
    
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        try:
            self.conn_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {str(e)}")
    
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
                # Convert the timestamp (milliseconds since epoch) to a proper timestamp
                timestamp_ms = order_data.get('updateTime') or order_data.get('time')
                if timestamp_ms is not None:
                    # Convert milliseconds to seconds and create a timestamp
                    transaction_time = datetime.fromtimestamp(timestamp_ms / 1000.0)
                else:
                    # Use current time if no timestamp is provided
                    transaction_time = datetime.now()
                
                cursor.execute("""
                    INSERT INTO executed_trades
                    (decision_id, exchange_order_id, trading_pair, side, 
                     order_type, status, requested_quantity, filled_quantity,
                     average_fill_price, commission, commission_asset, 
                     transaction_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    transaction_time  # Now using the converted timestamp
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
    
    async def log_order_events(self, events):
        """Log order book events to the database
        
        Args:
            events: List of order event dicts
            
        Returns:
            Number of events logged
        """
        if not events:
            return 0
            
        conn = self._get_connection()
        if not conn:
            return 0
            
        try:
            with conn.cursor() as cursor:
                # Bulk insert events
                values = []
                for event in events:
                    values.append((
                        event['trading_pair'],
                        event['synthetic_order_id'],
                        event['event_type'],
                        event['timestamp'],
                        event['price'],
                        event['quantity'],
                        event['side'],
                        event.get('remaining_quantity')
                    ))
                
                execute_values(
                    cursor,
                    """
                    INSERT INTO order_events
                    (trading_pair, synthetic_order_id, event_type, event_timestamp,
                     price, quantity, side, remaining_quantity)
                    VALUES %s
                    """,
                    values
                )
                
                # Update active orders based on these events
                for event in events:
                    if event['event_type'] == 'NEW':
                        # Insert new order
                        cursor.execute("""
                            INSERT INTO active_orders
                            (synthetic_order_id, trading_pair, first_seen, last_updated,
                             price, quantity, side, status)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (synthetic_order_id) DO NOTHING
                        """, (
                            event['synthetic_order_id'],
                            event['trading_pair'],
                            event['timestamp'],
                            event['timestamp'],
                            event['price'],
                            event['quantity'],
                            event['side'],
                            'ACTIVE'
                        ))
                    
                    elif event['event_type'] == 'MODIFY':
                        # Update existing order
                        cursor.execute("""
                            UPDATE active_orders
                            SET last_updated = %s, quantity = %s
                            WHERE synthetic_order_id = %s
                        """, (
                            event['timestamp'],
                            event['quantity'],
                            event['synthetic_order_id']
                        ))
                    
                    elif event['event_type'] in ('CANCEL', 'FILL'):
                        # Update order status
                        status = 'CANCELED' if event['event_type'] == 'CANCEL' else 'FILLED'
                        cursor.execute("""
                            UPDATE active_orders
                            SET last_updated = %s, status = %s
                            WHERE synthetic_order_id = %s
                        """, (
                            event['timestamp'],
                            status,
                            event['synthetic_order_id']
                        ))
                
                conn.commit()
                logger.debug(f"Logged {len(events)} order events")
                return len(events)
                
        except Exception as e:
            logger.error(f"Error logging order events: {str(e)}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                self._return_connection(conn)
    
    async def update_position_state(self, trading_pair, position, entry_price=None, 
                                  position_size=None, entry_timestamp=None):
        """Update the position state for a trading pair
        
        Args:
            trading_pair: The trading pair
            position: Position status ('LONG', 'SHORT', 'NONE')
            entry_price: Entry price (optional, for new positions)
            position_size: Position size (optional, for new positions) 
            entry_timestamp: Entry timestamp (optional, for new positions)
            
        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for position state update")
            return False
            
        try:
            logger.info(f"Updating position state in database: {trading_pair} -> {position}")
            logger.info(f"Position details: entry_price={entry_price}, size={position_size}, timestamp={entry_timestamp}")
            
            with conn.cursor() as cursor:
                # Check if record exists
                cursor.execute("""
                    SELECT id, position FROM position_state
                    WHERE trading_pair = %s
                """, (trading_pair,))
                
                result = cursor.fetchone()
                current_position = result[1] if result else None
                
                logger.info(f"Current position in database: {current_position}")
                
                if result:
                    # Update existing record
                    if position == 'NONE':
                        # Closing position
                        cursor.execute("""
                            UPDATE position_state
                            SET position = %s, 
                                entry_price = NULL,
                                position_size = NULL,
                                entry_timestamp = NULL,
                                holding_periods = 0,
                                last_update_timestamp = NOW()
                            WHERE trading_pair = %s
                        """, (position, trading_pair))
                        logger.info("Position closed in database")
                    else:
                        # Updating or entering position
                        cursor.execute("""
                            UPDATE position_state
                            SET position = %s, 
                                entry_price = COALESCE(%s, entry_price),
                                position_size = COALESCE(%s, position_size),
                                entry_timestamp = COALESCE(%s, entry_timestamp),
                                last_update_timestamp = NOW()
                            WHERE trading_pair = %s
                        """, (position, entry_price, position_size, entry_timestamp, trading_pair))
                        logger.info("Position updated in database")
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO position_state
                        (trading_pair, position, entry_price, position_size, entry_timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (trading_pair, position, entry_price, position_size, entry_timestamp))
                    logger.info("New position created in database")
                
                conn.commit()
                logger.info(f"Position state transaction committed for {trading_pair}: {position}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)
    
    async def get_position_state(self, trading_pair):
        """Get the current position state for a trading pair
        
        Args:
            trading_pair: The trading pair
            
        Returns:
            Dict with position state or None if not found
        """
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for position state query")
            return None
            
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT position, entry_price, position_size, 
                           entry_timestamp, holding_periods, last_update_timestamp
                    FROM position_state
                    WHERE trading_pair = %s
                """, (trading_pair,))
                
                row = cursor.fetchone()
                
                if row:
                    position_state = {
                        'position': row[0],
                        'entry_price': row[1],
                        'position_size': row[2],
                        'entry_timestamp': row[3],
                        'holding_periods': row[4],
                        'last_update': row[5]
                    }
                    logger.info(f"Retrieved position state from DB for {trading_pair}: {position_state['position']}")
                    logger.debug(f"Full position state: {position_state}")
                    return position_state
                else:
                    # No position found, return default state
                    logger.info(f"No position state found in DB for {trading_pair}, returning default NONE")
                    return {
                        'position': 'NONE',
                        'entry_price': None,
                        'position_size': None,
                        'entry_timestamp': None,
                        'holding_periods': 0,
                        'last_update': None
                    }
                
        except Exception as e:
            logger.error(f"Error getting position state: {str(e)}", exc_info=True)
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def increment_holding_period(self, trading_pair):
        """Increment the holding period counter for a position
        
        Args:
            trading_pair: The trading pair
            
        Returns:
            New holding period value or None if failed
        """
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE position_state
                    SET holding_periods = holding_periods + 1
                    WHERE trading_pair = %s AND position != 'NONE'
                    RETURNING holding_periods
                """, (trading_pair,))
                
                row = cursor.fetchone()
                
                conn.commit()
                return row[0] if row else 0
                
        except Exception as e:
            logger.error(f"Error incrementing holding period: {str(e)}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def log_trade(self, timestamp, trading_pair, direction, price, size, action, order_type, status, pnl=0.0, fees=0.0, order_id=None):
        """Log a trade to the database
        
        Args:
            timestamp: Timestamp of the trade
            trading_pair: The trading pair
            direction: Trade direction (BUY/SELL)
            price: Execution price
            size: Trade size
            action: OPEN or CLOSE
            order_type: Type of order (MARKET, LIMIT, etc.)
            status: Order status (FILLED, etc.)
            pnl: PnL realized (only for CLOSE trades)
            fees: Fees paid for the trade
            order_id: Optional order ID (not stored in database)
            
        Returns:
            Trade ID if successful, None otherwise
        """
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for trade logging")
            return None
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO trades
                    (timestamp, trading_pair, direction, price, size, action, order_type, status, pnl, fees)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    timestamp,
                    trading_pair,
                    direction,
                    price,
                    size,
                    action,
                    order_type,
                    status,
                    pnl if action == 'CLOSE' else 0.0,  # Only log PnL for close trades
                    fees
                ))
                
                trade_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Logged trade {trade_id}: {action} {direction} {size} {trading_pair} @ {price}")
                if action == 'CLOSE':
                    logger.info(f"Trade PnL: {pnl:.6f}, Fees: {fees:.6f}")
                    
                    # Update performance metrics table
                    try:
                        cursor.execute("""
                            INSERT INTO performance_metrics
                            (timestamp, trading_pair, trade_id, cumulative_pnl, trade_pnl, fees)
                            VALUES (%s, %s, %s, 
                                (SELECT COALESCE(MAX(cumulative_pnl), 0) FROM performance_metrics) + %s,
                                %s, %s)
                        """, (
                            timestamp,
                            trading_pair,
                            trade_id,
                            pnl - fees,  # Net PnL
                            pnl,
                            fees
                        ))
                        conn.commit()
                        logger.info(f"Updated performance metrics with trade {trade_id}")
                    except Exception as e:
                        logger.error(f"Error updating performance metrics: {str(e)}", exc_info=True)
                        conn.rollback()
                
                return trade_id
                
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self._return_connection(conn)
    
    async def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to get database connection for creating tables")
            return False
        
        try:
            with conn.cursor() as cursor:
                # Create predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        trading_pair VARCHAR(20) NOT NULL,
                        dir_prediction FLOAT NOT NULL,
                        down_prediction FLOAT NOT NULL,
                        ensemble_prediction FLOAT NOT NULL,
                        feature_context JSONB
                    )
                """)
                
                # Create signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        trading_pair VARCHAR(20) NOT NULL,
                        direction VARCHAR(10) NOT NULL,
                        strength FLOAT NOT NULL,
                        should_trade BOOLEAN NOT NULL,
                        current_price FLOAT
                    )
                """)
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        trading_pair VARCHAR(20) NOT NULL,
                        direction VARCHAR(10) NOT NULL,
                        price FLOAT NOT NULL,
                        size FLOAT NOT NULL,
                        action VARCHAR(10) NOT NULL,
                        order_type VARCHAR(20) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        pnl FLOAT,
                        fees FLOAT
                    )
                """)
                
                # Create position_state table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS position_state (
                        id SERIAL PRIMARY KEY,
                        trading_pair VARCHAR(20) UNIQUE NOT NULL,
                        position VARCHAR(10) NOT NULL DEFAULT 'NONE',
                        entry_price FLOAT,
                        position_size FLOAT,
                        entry_timestamp TIMESTAMP,
                        holding_periods INTEGER DEFAULT 0,
                        last_update_timestamp TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create trade_decisions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_decisions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        trading_pair VARCHAR(20) NOT NULL,
                        prediction_id INTEGER,
                        decision VARCHAR(20) NOT NULL,
                        target_price FLOAT,
                        prediction_value FLOAT,
                        reason TEXT
                    )
                """)
                
                # Create executed_trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS executed_trades (
                        id SERIAL PRIMARY KEY,
                        decision_id INTEGER NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        trade_data JSONB NOT NULL
                    )
                """)
                
                # Create order_book_events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS order_book_events (
                        id SERIAL PRIMARY KEY,
                        trading_pair VARCHAR(20) NOT NULL,
                        synthetic_order_id VARCHAR(50) NOT NULL,
                        event_type VARCHAR(10) NOT NULL,
                        event_timestamp TIMESTAMP NOT NULL,
                        price FLOAT NOT NULL,
                        quantity FLOAT NOT NULL,
                        side VARCHAR(4) NOT NULL,
                        remaining_quantity FLOAT
                    )
                """)
                
                # Create performance_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        trading_pair VARCHAR(20) NOT NULL,
                        trade_id INTEGER,
                        cumulative_pnl FLOAT NOT NULL,
                        trade_pnl FLOAT,
                        fees FLOAT,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        max_drawdown FLOAT
                    )
                """)
                
                conn.commit()
                logger.info("Database tables created/verified successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn) 