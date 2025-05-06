import logging
import time
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class RealtimeTrader:
    """
    Handles trade execution based on signals from the model
    """
    
    def __init__(self, config, db_logger=None):
        """Initialize the trader with configuration"""
        self.config = config
        self.db_logger = db_logger
        self.live_trading = config.get('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        self.simulation_mode = config.get('SIMULATION_MODE', 'false').lower() == 'true'
        self.trading_pair = config['TRADING_PAIR']
        self.exchange_symbol = config['EXCHANGE_SYMBOL']
        
        # Position sizing config
        self.position_sizing_enabled = config.get('POSITION_SIZING_ENABLED', False)
        self.position_size_base = float(config.get('POSITION_SIZE_BASE', 0.01))
        self.position_size_factor = float(config.get('POSITION_SIZE_FACTOR', 0.0))
        self.position_size_fixed = float(config.get('POSITION_SIZE_FIXED', 0.01))
        
        # Risk management config
        self.stop_loss = float(config.get('STOP_LOSS', 0.01))
        self.take_profit = float(config.get('TAKE_PROFIT', 0.02))
        self.max_holding_period = int(config.get('MAX_HOLDING_PERIOD', 8))
        
        # Trading fees
        self.fee_per_trade = float(config.get('FEE_PER_TRADE', 0.0001))
        
        # Current position info
        self.current_position = None
        self.position_history = []
        self.pnl_history = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = 1.0  # Assuming starting with 1 unit
        
        # Operational state
        self.last_trade_time = None
        self.min_time_between_trades = 300  # 5 minutes in seconds
        
        # Log configuration
        mode_str = "LIVE TRADING" if self.live_trading else "SIMULATION MODE"
        logger.info(f"RealtimeTrader initialized in {mode_str} for {self.trading_pair}")
        if self.position_sizing_enabled:
            logger.info(f"Position sizing enabled: base={self.position_size_base}, factor={self.position_size_factor}")
        logger.info(f"Risk parameters: SL={self.stop_loss}, TP={self.take_profit}, max holding={self.max_holding_period}")
    
    async def execute_trade(self, signal):
        """
        Execute a trade based on the signal
        
        Args:
            signal: Dict with signal data
            
        Returns:
            Dict with trade execution results
        """
        # Check if live trading is enabled
        if not self.live_trading and not self.simulation_mode:
            logger.info("Live trading is disabled. Not executing trade.")
            return {'status': 'skipped', 'reason': 'live_trading_disabled'}
        
        # Check if signal indicates a trade
        if not signal.get('should_trade', False):
            logger.info("Signal doesn't indicate a trade. Not executing.")
            return {'status': 'skipped', 'reason': 'signal_below_threshold'}
        
        # Check if enough time has passed since last trade
        if self.last_trade_time and time.time() - self.last_trade_time < self.min_time_between_trades:
            logger.info(f"Too soon since last trade. Waiting at least {self.min_time_between_trades}s between trades.")
            return {'status': 'skipped', 'reason': 'trade_frequency_limit'}
        
        # Determine trade direction
        direction = signal.get('signal_direction', 'NEUTRAL')
        if direction == 'NEUTRAL':
            logger.info("Neutral signal. Not executing trade.")
            return {'status': 'skipped', 'reason': 'neutral_signal'}
        
        # Check if we already have a position in the same direction
        if self.current_position and self.current_position['direction'] == direction:
            logger.info(f"Already have {direction} position. Not opening another.")
            return {'status': 'skipped', 'reason': 'position_exists'}
        
        # Close any existing opposite position
        if self.current_position and self.current_position['direction'] != direction:
            logger.info(f"Closing existing {self.current_position['direction']} position before opening {direction}.")
            close_result = await self._close_position()
            
            # If we failed to close, don't open a new position
            if close_result.get('status') != 'success':
                logger.warning("Failed to close existing position. Not opening new position.")
                return {'status': 'failed', 'reason': 'close_position_failed'}
        
        # Calculate position size based on signal strength
        position_size = self._calculate_position_size(signal)
        
        # Current price
        current_price = signal.get('current_price')
        if not current_price:
            logger.error("No current price in signal. Cannot execute trade.")
            return {'status': 'failed', 'reason': 'no_price_data'}
        
        # Calculate entry, stop loss, and take profit levels
        entry_price = current_price
        sl_price = self._calculate_stop_loss(direction, entry_price)
        tp_price = self._calculate_take_profit(direction, entry_price)
        
        # Create position object
        position = {
            'direction': direction,
            'entry_price': entry_price,
            'size': position_size,
            'entry_time': datetime.now(),
            'sl_price': sl_price,
            'tp_price': tp_price,
            'signal_strength': signal.get('signal_strength'),
            'holding_period': 0,
            'status': 'open'
        }
        
        # Execute trade - in simulation mode we just log it
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Opening {direction} position of size {position_size} at {entry_price}")
            self.current_position = position
            self.total_trades += 1
            self.last_trade_time = time.time()
            
            # Log to database if available
            if self.db_logger:
                await self.db_logger.log_trade(
                    timestamp=position['entry_time'],
                    trading_pair=self.trading_pair,
                    direction=direction,
                    price=entry_price,
                    size=position_size,
                    action='OPEN',
                    order_type='MARKET',
                    status='FILLED',
                    pnl=0.0,
                    fees=position_size * entry_price * self.fee_per_trade
                )
            
            return {
                'status': 'success',
                'action': 'open',
                'direction': direction,
                'size': position_size,
                'price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'time': position['entry_time']
            }
        elif self.live_trading:
            # Real trading via API would go here
            # For now, we'll just log the intention
            logger.info(f"[LIVE] Would open {direction} position of size {position_size} at {entry_price}")
            logger.warning("Live trading API integration not implemented yet")
            
            # Would need to call exchange API here
            # Example placeholder:
            # result = await self._call_exchange_api('open_position', {
            #     'symbol': self.exchange_symbol,
            #     'direction': 'BUY' if direction == 'BUY' else 'SELL',
            #     'quantity': position_size,
            #     'price': entry_price,
            #     'type': 'MARKET'
            # })
            
            # For now, simulate success
            self.current_position = position
            self.total_trades += 1
            self.last_trade_time = time.time()
            
            return {
                'status': 'success',
                'action': 'open',
                'direction': direction,
                'size': position_size,
                'price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'time': position['entry_time']
            }
    
    async def _close_position(self, reason='signal'):
        """
        Close the current position
        
        Args:
            reason: Reason for closing the position ('signal', 'stop_loss', 'take_profit', 'max_holding')
            
        Returns:
            Dict with close position results
        """
        if not self.current_position:
            logger.info("No position to close.")
            return {'status': 'skipped', 'reason': 'no_position'}
        
        position = self.current_position
        direction = position['direction']
        entry_price = position['entry_price']
        size = position['size']
        entry_time = position['entry_time']
        
        # Get current price - in real implementation this would come from API
        # For simulation, we'll just use the last price + estimated slippage
        current_price = entry_price * (0.999 if direction == 'BUY' else 1.001)  # Simulate slippage
        
        # Calculate PnL
        pnl = self._calculate_pnl(direction, entry_price, current_price, size)
        
        # Calculate fees
        fees = size * current_price * self.fee_per_trade
        
        # Net PnL
        net_pnl = pnl - fees
        
        # Update performance metrics
        self.total_pnl += net_pnl
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Track drawdown
        current_value = 1.0 + self.total_pnl  # Assuming starting with 1 unit
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Store position in history
        position['exit_price'] = current_price
        position['exit_time'] = datetime.now()
        position['pnl'] = net_pnl
        position['fees'] = fees
        position['holding_period'] = (position['exit_time'] - entry_time).total_seconds() / 60  # in minutes
        position['status'] = 'closed'
        position['close_reason'] = reason
        
        self.position_history.append(position)
        self.pnl_history.append(net_pnl)
        
        # Execute close - in simulation mode we just log it
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Closing {direction} position of size {size} at {current_price}. " +
                        f"PnL: {net_pnl:.6f}, Reason: {reason}")
            
            # Log to database if available
            if self.db_logger:
                await self.db_logger.log_trade(
                    timestamp=position['exit_time'],
                    trading_pair=self.trading_pair,
                    direction='SELL' if direction == 'BUY' else 'BUY',  # Opposite direction to close
                    price=current_price,
                    size=size,
                    action='CLOSE',
                    order_type='MARKET',
                    status='FILLED',
                    pnl=net_pnl,
                    fees=fees
                )
            
            # Clear current position
            self.current_position = None
            
            return {
                'status': 'success',
                'action': 'close',
                'direction': direction,
                'size': size,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': net_pnl,
                'fees': fees,
                'reason': reason,
                'time': position['exit_time']
            }
        elif self.live_trading:
            # Real trading via API would go here
            # For now, we'll just log the intention
            logger.info(f"[LIVE] Would close {direction} position of size {size} at market price. Reason: {reason}")
            logger.warning("Live trading API integration not implemented yet")
            
            # Would need to call exchange API here
            # Example placeholder:
            # result = await self._call_exchange_api('close_position', {
            #     'symbol': self.exchange_symbol,
            #     'direction': 'SELL' if direction == 'BUY' else 'BUY',  # Opposite to close
            #     'quantity': size,
            #     'type': 'MARKET'
            # })
            
            # For now, simulate success
            # Clear current position
            self.current_position = None
            
            return {
                'status': 'success',
                'action': 'close',
                'direction': direction,
                'size': size,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': net_pnl,
                'fees': fees,
                'reason': reason,
                'time': position['exit_time']
            }
    
    def _calculate_position_size(self, signal):
        """
        Calculate position size based on signal strength and configuration
        
        Args:
            signal: Dict with signal data
            
        Returns:
            Position size
        """
        if not self.position_sizing_enabled:
            return self.position_size_fixed
            
        # Get signal strength
        signal_strength = abs(signal.get('signal_strength', 0))
        
        # Calculate size based on signal strength
        size = self.position_size_base + (signal_strength * self.position_size_factor)
        
        # Ensure minimum size
        return max(size, self.position_size_fixed)
    
    def _calculate_stop_loss(self, direction, entry_price):
        """Calculate stop loss price based on direction and entry price"""
        if direction == 'BUY':
            return entry_price * (1 - self.stop_loss)
        else:
            return entry_price * (1 + self.stop_loss)
    
    def _calculate_take_profit(self, direction, entry_price):
        """Calculate take profit price based on direction and entry price"""
        if direction == 'BUY':
            return entry_price * (1 + self.take_profit)
        else:
            return entry_price * (1 - self.take_profit)
    
    def _calculate_pnl(self, direction, entry_price, exit_price, size):
        """
        Calculate profit/loss for a position
        
        Args:
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            
        Returns:
            PnL amount
        """
        if direction == 'BUY':
            return size * (exit_price - entry_price)
        else:
            return size * (entry_price - exit_price)
    
    async def check_positions(self, current_price):
        """
        Check current positions for stop loss, take profit, or max holding period
        
        Args:
            current_price: Current price for the trading pair
            
        Returns:
            Dict with position check results
        """
        if not self.current_position:
            return {'status': 'skipped', 'reason': 'no_position'}
            
        position = self.current_position
        direction = position['direction']
        entry_price = position['entry_price']
        sl_price = position['sl_price']
        tp_price = position['tp_price']
        holding_period = position['holding_period']
        
        # Increment holding period
        position['holding_period'] = holding_period + 1
        
        # Check stop loss
        if (direction == 'BUY' and current_price <= sl_price) or \
           (direction == 'SELL' and current_price >= sl_price):
            logger.info(f"Stop loss triggered at {current_price}. Closing position.")
            return await self._close_position(reason='stop_loss')
            
        # Check take profit
        if (direction == 'BUY' and current_price >= tp_price) or \
           (direction == 'SELL' and current_price <= tp_price):
            logger.info(f"Take profit triggered at {current_price}. Closing position.")
            return await self._close_position(reason='take_profit')
            
        # Check max holding period
        if holding_period >= self.max_holding_period:
            logger.info(f"Max holding period ({self.max_holding_period}) reached. Closing position.")
            return await self._close_position(reason='max_holding')
            
        # Position still open
        return {'status': 'open', 'position': position}
    
    def get_position_status(self):
        """Get current position status"""
        if not self.current_position:
            return {'status': 'no_position'}
            
        return {
            'status': 'open',
            'position': self.current_position
        }
    
    def get_performance_metrics(self):
        """Get trading performance metrics"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio() if len(self.pnl_history) > 10 else None,
            'current_position': self.current_position
        }
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from PnL history"""
        if not self.pnl_history or len(self.pnl_history) < 2:
            return None
            
        import numpy as np
        returns = np.array(self.pnl_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return None
            
        # Assuming risk-free rate of 0 and annualization factor of sqrt(365)
        sharpe = (mean_return / std_return) * (365 ** 0.5)
        return sharpe 