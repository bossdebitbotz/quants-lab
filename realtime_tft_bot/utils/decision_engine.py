import logging
import time
import numpy as np
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Implements trading logic based on TFT model predictions
    """
    
    def __init__(self, config, db_logger):
        self.config = config
        self.db_logger = db_logger
        self.trading_pair = config['TRADING_PAIR']
        self.trading_threshold = config['TRADING_THRESHOLD']
        self.position_sizing_enabled = config['POSITION_SIZING_ENABLED']
        self.position_size_base = config['POSITION_SIZE_BASE']
        self.position_size_factor = config['POSITION_SIZE_FACTOR']
        self.position_size_fixed = config['POSITION_SIZE_FIXED']
        self.stop_loss = config['STOP_LOSS']
        self.take_profit = config['TAKE_PROFIT']
        self.max_holding_period = config['MAX_HOLDING_PERIOD']
        self.use_dynamic_threshold = config['USE_DYNAMIC_THRESHOLD']
        self.enable_live_trading = config['ENABLE_LIVE_TRADING']
        self.last_decision_time = None
        self.current_position = None
        self.dynamic_threshold = self.trading_threshold  # Start with configured value
        self.recent_predictions = []  # Store recent predictions for dynamic threshold
        self.position_lock = asyncio.Lock()  # Lock for position operations
        
    async def initialize(self):
        """Initialize decision engine"""
        # Load current position state from database
        self.current_position = await self.db_logger.get_position_state(self.trading_pair)
        if not self.current_position:
            # If no position found, set to default NONE
            self.current_position = {
                'position': 'NONE',
                'entry_price': None,
                'position_size': None,
                'entry_timestamp': None,
                'holding_periods': 0
            }
            
        logger.info(f"Decision engine initialized. Current position: {self.current_position['position']}")
        logger.info(f"Initial position details: {self.current_position}")
        return True
        
    async def reload_position_state(self):
        """Reload position state from database to ensure consistency"""
        logger.info("Reloading position state from database")
        self.current_position = await self.db_logger.get_position_state(self.trading_pair)
        if not self.current_position:
            self.current_position = {
                'position': 'NONE',
                'entry_price': None,
                'position_size': None,
                'entry_timestamp': None,
                'holding_periods': 0
            }
        logger.info(f"Reloaded position state: {self.current_position['position']}")
        logger.info(f"Current position details: {self.current_position}")
        return self.current_position
        
    async def make_decision(self, prediction, current_price):
        """
        Make a trading decision based on the current prediction and market state
        
        Args:
            prediction: Dict with prediction data
                - dir_prediction: Directional model prediction
                - down_prediction: Downward specialist prediction
                - ensemble_prediction: Ensemble prediction
                - timestamp: Prediction timestamp
            current_price: Current market price
            
        Returns:
            Dict with decision details
        """
        if prediction is None:
            logger.warning("No valid prediction provided to decision engine")
            return self._create_decision('HOLD', None, "No valid prediction")
            
        # Re-check position state from database to ensure consistency
        db_position = await self.db_logger.get_position_state(self.trading_pair)
        if db_position and db_position['position'] != self.current_position['position']:
            logger.warning(f"Position state inconsistency detected. DB: {db_position['position']}, " 
                          f"Local: {self.current_position['position']}. Updating local state.")
            self.current_position = db_position
            logger.info(f"Updated position details: {self.current_position}")
        
        # Store prediction for dynamic threshold calculation
        self.recent_predictions.append({
            'value': prediction['ensemble_prediction'],
            'timestamp': prediction['timestamp']
        })
        
        # Clean up old predictions (keep last 100)
        if len(self.recent_predictions) > 100:
            self.recent_predictions = self.recent_predictions[-100:]
        
        # Update dynamic threshold if enabled
        if self.use_dynamic_threshold:
            self._update_dynamic_threshold()
        
        # Get current trading threshold
        current_threshold = self.dynamic_threshold if self.use_dynamic_threshold else self.trading_threshold
        
        # Get prediction value
        pred_value = prediction['ensemble_prediction']
        
        # Get current position
        position = self.current_position['position']
        
        logger.info(f"Making decision with prediction value: {pred_value:.6f}, " 
                    f"threshold: {current_threshold:.6f}, position: {position}")
        
        # Process existing position if we have one
        if position != 'NONE':
            logger.info(f"Processing existing {position} position")
            decision = await self._process_existing_position(position, pred_value, current_price)
            if decision:
                return decision
        
        # Process new potential position
        logger.info("No position or keeping existing position, checking for new position signals")
        decision = await self._process_new_position(pred_value, current_threshold, current_price)
        logger.info(f"Decision: {decision['decision']}, Reason: {decision.get('reason')}")
        return decision
    
    async def _process_existing_position(self, position, pred_value, current_price):
        """
        Process an existing position to determine if it should be exited
        
        Args:
            position: Current position type ('LONG' or 'SHORT')
            pred_value: Prediction value
            current_price: Current market price
            
        Returns:
            Decision dict or None if position should be kept
        """
        # Skip if no entry price (shouldn't happen but just in case)
        if not self.current_position['entry_price']:
            logger.warning(f"Position {position} has no entry price. Setting to NONE.")
            await self.db_logger.update_position_state(self.trading_pair, 'NONE')
            self.current_position['position'] = 'NONE'
            return None
            
        entry_price = self.current_position['entry_price']
        entry_time = self.current_position['entry_timestamp']
        holding_periods = self.current_position['holding_periods']
        
        # Calculate current P&L
        pnl = 0
        if position == 'LONG':
            pnl = (float(current_price) / float(entry_price)) - 1
        elif position == 'SHORT':
            pnl = 1 - (float(current_price) / float(entry_price))
        
        logger.info(f"Evaluating {position} position: entry price: {entry_price}, " 
                   f"current price: {current_price}, P&L: {pnl:.4f}, periods: {holding_periods}")
        
        # Check for exit conditions
        reason = None
        
        # 1. Check stop loss
        if position == 'LONG' and pnl < -self.stop_loss:
            reason = f"Stop loss triggered: {pnl:.4f} < -{self.stop_loss:.4f}"
        elif position == 'SHORT' and pnl < -self.stop_loss:
            reason = f"Stop loss triggered: {pnl:.4f} < -{self.stop_loss:.4f}"
            
        # 2. Check take profit
        elif position == 'LONG' and pnl > self.take_profit:
            reason = f"Take profit triggered: {pnl:.4f} > {self.take_profit:.4f}"
        elif position == 'SHORT' and pnl > self.take_profit:
            reason = f"Take profit triggered: {pnl:.4f} > {self.take_profit:.4f}"
            
        # 3. Check holding period
        elif holding_periods >= self.max_holding_period:
            reason = f"Max holding period reached: {holding_periods} >= {self.max_holding_period}"
            
        # 4. Check for reversal signal
        elif (position == 'LONG' and pred_value < -self.trading_threshold):
            reason = f"Reversal signal: Prediction {pred_value:.6f} < -{self.trading_threshold:.6f}"
        elif (position == 'SHORT' and pred_value > self.trading_threshold):
            reason = f"Reversal signal: Prediction {pred_value:.6f} > {self.trading_threshold:.6f}"
        
        # If any exit condition triggered, exit position
        if reason:
            # Create exit decision
            decision_type = 'EXIT_LONG' if position == 'LONG' else 'EXIT_SHORT'
            decision = self._create_decision(decision_type, pred_value, reason, current_price, pnl)
            
            # Update position state
            logger.info(f"Exiting {position} position. Reason: {reason}. P&L: {pnl:.4f}")
            await self.db_logger.update_position_state(self.trading_pair, 'NONE')
            self.current_position = {
                'position': 'NONE',
                'entry_price': None,
                'position_size': None,
                'entry_timestamp': None,
                'holding_periods': 0
            }
            
            return decision
            
        # Otherwise, hold position
        logger.info(f"Maintaining {position} position. Current P&L: {pnl:.4f}")
        return self._create_decision('HOLD', pred_value, f"Maintaining {position} position. Current P&L: {pnl:.4f}")
    
    async def _process_new_position(self, pred_value, threshold, current_price):
        """
        Process potential new position entry
        
        Args:
            pred_value: Prediction value
            threshold: Current trading threshold
            current_price: Current market price
            
        Returns:
            Decision dict
        """
        # Acquire the position lock to prevent race conditions
        async with self.position_lock:
            # Check position state again after acquiring lock
            current_db_position = await self.db_logger.get_position_state(self.trading_pair)
            if current_db_position and current_db_position['position'] != 'NONE':
                logger.info(f"Position changed while waiting for lock. Current position: {current_db_position['position']}")
                self.current_position = current_db_position
                return self._create_decision('HOLD', pred_value, f"Position already {current_db_position['position']}")
                
            # Check for entry signals
            if pred_value > threshold:
                # Long signal
                size = self._calculate_position_size(pred_value, threshold)
                decision = self._create_decision('ENTER_LONG', pred_value, 
                    f"Strong upward signal: {pred_value:.6f} > {threshold:.6f}", current_price)
                
                # Update position state if live trading enabled
                if self.enable_live_trading:
                    logger.info(f"Entering LONG position at {current_price} with size {size}")
                    await self.db_logger.update_position_state(
                        self.trading_pair,
                        'LONG',
                        entry_price=current_price,
                        position_size=size,
                        entry_timestamp=datetime.now()
                    )
                    self.current_position = {
                        'position': 'LONG',
                        'entry_price': current_price,
                        'position_size': size,
                        'entry_timestamp': datetime.now(),
                        'holding_periods': 0
                    }
                else:
                    logger.info(f"Simulated LONG entry signal at {current_price} with size {size} (live trading disabled)")
                    
                return decision
                
            elif pred_value < -threshold:
                # Short signal
                size = self._calculate_position_size(abs(pred_value), threshold)
                decision = self._create_decision('ENTER_SHORT', pred_value, 
                    f"Strong downward signal: {pred_value:.6f} < -{threshold:.6f}", current_price)
                
                # Update position state if live trading enabled
                if self.enable_live_trading:
                    logger.info(f"Entering SHORT position at {current_price} with size {size}")
                    await self.db_logger.update_position_state(
                        self.trading_pair,
                        'SHORT',
                        entry_price=current_price,
                        position_size=size,
                        entry_timestamp=datetime.now()
                    )
                    self.current_position = {
                        'position': 'SHORT',
                        'entry_price': current_price,
                        'position_size': size,
                        'entry_timestamp': datetime.now(),
                        'holding_periods': 0
                    }
                else:
                    logger.info(f"Simulated SHORT entry signal at {current_price} with size {size} (live trading disabled)")
                    
                return decision
                
            # No signal, hold
            logger.info(f"No trading signal: {pred_value:.6f}, threshold: {threshold:.6f}")
            return self._create_decision('HOLD', pred_value, f"No trading signal: {pred_value:.6f}, threshold: {threshold:.6f}")
    
    def _calculate_position_size(self, signal_strength, threshold):
        """
        Calculate position size based on signal strength
        
        Args:
            signal_strength: Absolute prediction value
            threshold: Current trading threshold
            
        Returns:
            Position size (0.0 to 1.0)
        """
        if not self.position_sizing_enabled:
            return self.position_size_fixed
            
        # Calculate position size based on signal strength relative to threshold
        # with base size and scaling factor
        relative_strength = min(signal_strength / (threshold * 5), 1.0)  # Cap at 1.0
        size = self.position_size_base + (relative_strength * self.position_size_factor)
        
        # Ensure size is within bounds
        size = max(0.01, min(size, 1.0))
        
        logger.info(f"Calculated position size: {size:.4f} (strength: {signal_strength:.6f}, threshold: {threshold:.6f})")
        return size
    
    def _update_dynamic_threshold(self):
        """Update dynamic threshold based on recent prediction volatility"""
        if len(self.recent_predictions) < 20:
            return  # Not enough data
            
        # Get prediction values and calculate volatility
        values = [p['value'] for p in self.recent_predictions]
        volatility = np.std(values)
        
        # Scale threshold based on volatility
        # Higher volatility = higher threshold to reduce false signals
        base_threshold = self.trading_threshold
        if volatility > 0:
            # Adjust threshold based on volatility
            # This is a simple linear scaling, could be more sophisticated
            scaling_factor = 1.0 + (volatility / 0.002)  # Normalize to expected volatility
            scaling_factor = min(max(scaling_factor, 0.8), 2.0)  # Keep within reasonable bounds
            
            self.dynamic_threshold = base_threshold * scaling_factor
            logger.debug(f"Updated dynamic threshold: {self.dynamic_threshold:.6f} (volatility: {volatility:.6f})")
        else:
            self.dynamic_threshold = base_threshold
    
    async def increment_holding_period(self):
        """
        Increment holding period for current position
        
        Returns:
            New holding period or None if no active position
        """
        if self.current_position['position'] == 'NONE':
            return None
            
        # Increment in database
        new_periods = await self.db_logger.increment_holding_period(self.trading_pair)
        
        # Update local state
        if new_periods is not None:
            self.current_position['holding_periods'] = new_periods
            logger.debug(f"Incremented holding period for {self.trading_pair} to {new_periods}")
            
        return new_periods
    
    def _create_decision(self, decision_type, pred_value, reason, current_price=None, pnl=None):
        """
        Create a decision object
        
        Args:
            decision_type: Type of decision
            pred_value: Prediction value
            reason: Reason for the decision
            current_price: Current price (optional)
            pnl: Profit/loss (optional)
            
        Returns:
            Decision dict
        """
        decision = {
            'timestamp': datetime.now(),
            'decision': decision_type,
            'prediction_value': pred_value,
            'reason': reason,
            'current_price': current_price,
            'trading_pair': self.trading_pair
        }
        
        if pnl is not None:
            decision['pnl'] = pnl
            
        self.last_decision_time = decision['timestamp']
        logger.info(f"Created decision: {decision_type}, reason: {reason}")
        return decision 