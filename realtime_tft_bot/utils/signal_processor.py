import logging
import time
import asyncio
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Processes model predictions and generates trading signals
    """
    
    def __init__(self, config, db_logger=None):
        """Initialize signal processor with configuration"""
        self.config = config
        self.db_logger = db_logger
        self.threshold = config.get('TRADING_THRESHOLD', 0.0001)
        self.ensemble_method = config.get('ENSEMBLE_METHOD', 'selective')
        self.ensemble_threshold = config.get('ENSEMBLE_THRESHOLD', 0.002)
        self.dynamic_threshold = config.get('USE_DYNAMIC_THRESHOLD', False)
        
        # Signal tracking for validation
        self.signal_history = []
        self.max_history = 100
        
        logger.info(f"Signal processor initialized with threshold: {self.threshold}, ensemble method: {self.ensemble_method}")
    
    async def process_prediction(self, prediction):
        """
        Process model prediction and generate trading signal
        
        Args:
            prediction: Dict with prediction data
            
        Returns:
            Dict with trading signal data
        """
        try:
            # Extract prediction values
            dir_pred = prediction['dir_prediction']
            down_pred = prediction.get('down_prediction', 0)
            ensemble_pred = prediction.get('ensemble_prediction', dir_pred)
            
            # Current price info
            current_price = prediction.get('last_price')
            
            # Determine signal direction and strength
            signal_direction = self._determine_signal_direction(dir_pred, down_pred)
            signal_strength = self._calculate_signal_strength(dir_pred, down_pred)
            
            # Apply threshold to determine if we should trade
            should_trade = abs(signal_strength) > self.threshold
            
            # Create signal
            signal = {
                'timestamp': datetime.now(),
                'trading_pair': self.config.get('TRADING_PAIR'),
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'should_trade': should_trade,
                'current_price': current_price,
                'directional_prediction': dir_pred,
                'downward_prediction': down_pred,
                'ensemble_prediction': ensemble_pred,
                'threshold': self.threshold
            }
            
            # Log signal to database if available
            if self.db_logger:
                await self.db_logger.log_signal(
                    timestamp=signal['timestamp'],
                    trading_pair=signal['trading_pair'],
                    direction=signal['signal_direction'],
                    strength=signal['signal_strength'],
                    should_trade=signal['should_trade'],
                    current_price=signal['current_price']
                )
            
            # Add to history
            self._add_to_history(signal)
            
            logger.info(f"Signal processed: {signal_direction} with strength {signal_strength:.6f}" + 
                        f" ({'TRADE' if should_trade else 'HOLD'})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing prediction: {type(e).__name__} - {str(e)}")
            return None
    
    def _determine_signal_direction(self, dir_pred, down_pred):
        """
        Determine signal direction based on predictions
        
        Args:
            dir_pred: Directional prediction (-1 to 1)
            down_pred: Downward specialist prediction (0 to 1)
            
        Returns:
            'BUY', 'SELL', or 'NEUTRAL'
        """
        # Using ensemble method based on config
        if self.ensemble_method == 'selective':
            # Down specialist takes precedence for downward moves
            if down_pred > self.ensemble_threshold:
                return 'SELL'
            elif dir_pred > self.threshold:
                return 'BUY'
            elif dir_pred < -self.threshold:
                return 'SELL'
            else:
                return 'NEUTRAL'
                
        elif self.ensemble_method == 'average':
            # Simple average of models
            combined = (dir_pred - down_pred) / 2
            if combined > self.threshold:
                return 'BUY'
            elif combined < -self.threshold:
                return 'SELL'
            else:
                return 'NEUTRAL'
                
        elif self.ensemble_method == 'directional_only':
            # Only use directional model
            if dir_pred > self.threshold:
                return 'BUY'
            elif dir_pred < -self.threshold:
                return 'SELL'
            else:
                return 'NEUTRAL'
        
        # Default fallback
        return 'NEUTRAL'
    
    def _calculate_signal_strength(self, dir_pred, down_pred):
        """
        Calculate signal strength based on predictions
        
        Args:
            dir_pred: Directional prediction
            down_pred: Downward specialist prediction
            
        Returns:
            Signal strength (-1 to 1)
        """
        # Using ensemble method based on config
        if self.ensemble_method == 'selective':
            # Down specialist takes precedence for downward moves
            if down_pred > self.ensemble_threshold:
                return -down_pred  # Negative because it's a sell signal
            else:
                return dir_pred
                
        elif self.ensemble_method == 'average':
            # Simple average of models
            return (dir_pred - down_pred) / 2
                
        elif self.ensemble_method == 'directional_only':
            # Only use directional model
            return dir_pred
        
        # Default fallback
        return dir_pred
    
    def _add_to_history(self, signal):
        """Add signal to history and limit size"""
        self.signal_history.append(signal)
        
        # Limit history size
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history:]
    
    def get_signal_history(self):
        """Get signal history"""
        return self.signal_history
    
    def update_threshold(self, new_threshold):
        """Update trading threshold"""
        self.threshold = new_threshold
        logger.info(f"Signal threshold updated to {self.threshold}")
    
    def get_validation_metrics(self):
        """
        Calculate validation metrics for recent signals
        
        Returns:
            Dict with validation metrics
        """
        if len(self.signal_history) < 10:
            return {'error': 'Not enough signal history for validation'}
            
        # Calculate basic metrics
        signals = self.signal_history
        trade_signals = [s for s in signals if s['should_trade']]
        buy_signals = [s for s in signals if s['signal_direction'] == 'BUY']
        sell_signals = [s for s in signals if s['signal_direction'] == 'SELL']
        
        metrics = {
            'total_signals': len(signals),
            'trade_signals': len(trade_signals),
            'trade_ratio': len(trade_signals) / len(signals) if signals else 0,
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'buy_sell_ratio': len(buy_signals) / len(sell_signals) if sell_signals else float('inf'),
            'avg_strength': np.mean([abs(s['signal_strength']) for s in signals]) if signals else 0,
            'max_strength': max([abs(s['signal_strength']) for s in signals]) if signals else 0,
        }
        
        return metrics 