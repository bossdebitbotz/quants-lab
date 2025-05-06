import logging
import asyncio
import json
import time
import hmac
import hashlib
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """
    Handles execution of trading decisions via Binance API
    """
    
    def __init__(self, config, db_logger):
        self.config = config
        self.db_logger = db_logger
        self.trading_pair = config['TRADING_PAIR']
        self.exchange_symbol = config['EXCHANGE_SYMBOL']
        self.api_key = config['BINANCE_API_KEY']
        self.api_secret = config['BINANCE_API_SECRET']
        self.base_url = config['BINANCE_BASE_URL']
        self.enable_live_trading = config['ENABLE_LIVE_TRADING']
        self.session = None
        self.last_order_time = None
        self.min_order_interval = 5  # Minimum seconds between orders
        
    async def initialize(self):
        """Initialize execution handler"""
        if not self.enable_live_trading:
            logger.info("Live trading is disabled. Execution handler will simulate trades only.")
            return True
            
        if not self.api_key or not self.api_secret:
            logger.error("API credentials missing. Cannot initialize execution handler for live trading.")
            return False
            
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Set position mode to ONE_WAY (BOTH) for Binance Futures
        if 'fapi' in self.base_url:
            try:
                # Set position mode endpoint
                position_mode_endpoint = '/fapi/v1/positionSide/dual'
                
                # Prepare parameters
                params = {
                    'dualSidePosition': 'false',  # false means ONE_WAY mode (BOTH)
                    'timestamp': int(time.time() * 1000)
                }
                
                # Generate signature
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['signature'] = signature
                
                # Add API key to headers
                headers = {
                    'X-MBX-APIKEY': self.api_key
                }
                
                # Set position mode
                url = f"{self.base_url}{position_mode_endpoint}"
                logger.info("Setting position mode to ONE_WAY (BOTH)")
                
                async with self.session.post(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        logger.info("Position mode set to ONE_WAY (BOTH) successfully")
                    elif 'code' in response_data and response_data['code'] == -4059:
                        # Already in this mode
                        logger.info("Position mode already set to ONE_WAY (BOTH)")
                    else:
                        logger.warning(f"Failed to set position mode: {response_data}")
                        
            except Exception as e:
                logger.warning(f"Error setting position mode: {str(e)}")
                
        logger.info("Execution handler initialized")
        return True
        
    async def close(self):
        """Close connections"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Execution handler connections closed")
    
    async def execute_decision(self, decision):
        """
        Execute a trading decision
        
        Args:
            decision: Dict with decision details
                - decision: Type of decision (ENTER_LONG, ENTER_SHORT, EXIT_LONG, EXIT_SHORT, HOLD)
                - prediction_value: Prediction value
                - reason: Reason for the decision
                - current_price: Current market price
                - trading_pair: Trading pair
                
        Returns:
            Dict with execution results or None if decision is HOLD
        """
        decision_type = decision['decision']
        
        # Skip HOLD decisions
        if decision_type == 'HOLD':
            return None
            
        # For paper trading mode, simulate execution
        if not self.enable_live_trading:
            return await self._simulate_execution(decision)
            
        # Rate limit check
        current_time = time.time()
        if self.last_order_time and current_time - self.last_order_time < self.min_order_interval:
            wait_time = self.min_order_interval - (current_time - self.last_order_time)
            logger.info(f"Rate limiting: Waiting {wait_time:.2f}s before executing order")
            await asyncio.sleep(wait_time)
        
        # Map decision to order parameters
        if decision_type == 'ENTER_LONG':
            # Market buy order
            return await self._place_order('BUY', 'MARKET', decision)
            
        elif decision_type == 'ENTER_SHORT':
            # Market sell order
            return await self._place_order('SELL', 'MARKET', decision)
            
        elif decision_type == 'EXIT_LONG':
            # Market sell order to close long
            return await self._place_order('SELL', 'MARKET', decision)
            
        elif decision_type == 'EXIT_SHORT':
            # Market buy order to close short
            return await self._place_order('BUY', 'MARKET', decision)
            
        else:
            logger.warning(f"Unknown decision type: {decision_type}")
            return None
    
    async def _place_order(self, side, order_type, decision):
        """
        Place an order via Binance API
        
        Args:
            side: 'BUY' or 'SELL'
            order_type: 'MARKET' or 'LIMIT'
            decision: Decision dict with details
            
        Returns:
            Dict with order execution results or error
        """
        if not self.session:
            logger.error("HTTP session not initialized. Call initialize() first.")
            return {'error': 'Session not initialized'}
            
        # Get current price from decision
        current_price = decision.get('current_price', 1.0)
        
        # Calculate quantity to meet minimum notional value of $10 with 10x leverage
        # Using $10 for minimum order value with some buffer
        min_notional = 10.0
        leverage = 10.0
        
        # With leverage, we can buy more with the same capital
        effective_quantity = (min_notional * leverage) / current_price
        quantity = round(effective_quantity, 0)  # Round to whole number for WLD-USDT
        formatted_quantity = '{:.0f}'.format(quantity)  # No decimal places
        
        logger.info(f"Calculated order quantity: {formatted_quantity} at price {current_price} with {leverage}x leverage")
        
        # For live trading, place the actual order
        if self.enable_live_trading:
            try:
                # First, try to set leverage for the symbol
                leverage_endpoint = '/fapi/v1/leverage'
                if 'fapi' in self.base_url:
                    leverage_params = {
                        'symbol': self.exchange_symbol,
                        'leverage': int(leverage),
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    # Generate signature for leverage request
                    leverage_query_string = '&'.join([f"{k}={v}" for k, v in leverage_params.items()])
                    leverage_signature = hmac.new(
                        self.api_secret.encode('utf-8'),
                        leverage_query_string.encode('utf-8'),
                        hashlib.sha256
                    ).hexdigest()
                    leverage_params['signature'] = leverage_signature
                    
                    # Add API key to headers
                    headers = {
                        'X-MBX-APIKEY': self.api_key
                    }
                    
                    # Set leverage
                    leverage_url = f"{self.base_url}{leverage_endpoint}"
                    logger.info(f"Setting leverage to {leverage}x for {self.exchange_symbol}")
                    
                    try:
                        async with self.session.post(leverage_url, params=leverage_params, headers=headers) as response:
                            if response.status == 200:
                                logger.info(f"Leverage set successfully to {leverage}x")
                            else:
                                logger.warning(f"Failed to set leverage: {await response.text()}")
                    except Exception as e:
                        logger.warning(f"Error setting leverage: {str(e)}")
                
                # Create Binance API endpoint path
                endpoint = '/fapi/v1/order' if 'fapi' in self.base_url else '/api/v3/order'
                
                # Prepare parameters for the order
                params = {
                    'symbol': self.exchange_symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': formatted_quantity,
                    'timestamp': int(time.time() * 1000)
                }
                
                # For futures, specify position side
                if 'fapi' in self.base_url:
                    # Add positionSide parameter for hedge mode
                    # For BOTH mode, don't include this parameter
                    # params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
                    
                    # Alternatively, set reduceOnly for closing positions
                    if decision['decision'] in ('EXIT_LONG', 'EXIT_SHORT'):
                        params['reduceOnly'] = 'true'
                
                # Add recvWindow parameter to avoid timestamp issues
                params['recvWindow'] = 5000
                
                # Generate signature
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['signature'] = signature
                
                # Add API key to headers
                headers = {
                    'X-MBX-APIKEY': self.api_key
                }
                
                # Make the API request
                url = f"{self.base_url}{endpoint}"
                logger.info(f"Placing order via Binance API: {side} {formatted_quantity} {self.trading_pair}")
                
                async with self.session.post(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        logger.info(f"Order placed successfully: {response_data}")
                        
                        # Update last order time
                        self.last_order_time = time.time()
                        
                        # Log to database
                        decision_id = await self._log_decision(decision)
                        if decision_id:
                            await self.db_logger.log_executed_trade(decision_id, response_data)
                        
                        return response_data
                    else:
                        logger.error(f"Order placement failed: {response_data}")
                        return {'error': response_data}
                        
            except Exception as e:
                logger.error(f"Error placing order: {str(e)}")
                return {'error': str(e)}
            
        # For simulation mode or if live trading fails, return simulated response
        logger.info(f"Simulating {order_type} {side} order for {formatted_quantity} {self.trading_pair}")
        
        # Update last order time
        self.last_order_time = time.time()
        
        # Simulate a successful response
        simulated_order = {
            'orderId': f"sim_{int(time.time())}",
            'symbol': self.exchange_symbol,
            'side': side,
            'type': order_type,
            'status': 'FILLED',
            'origQty': formatted_quantity,
            'executedQty': formatted_quantity,
            'avgPrice': decision.get('current_price', 0),
            'commission': 0.0001 * float(formatted_quantity) * decision.get('current_price', 0),
            'commissionAsset': 'USDT',
            'time': int(time.time() * 1000)
        }
        
        # Log to database
        decision_id = await self._log_decision(decision)
        if decision_id:
            await self.db_logger.log_executed_trade(decision_id, simulated_order)
        
        return simulated_order
    
    async def _simulate_execution(self, decision):
        """
        Simulate execution for paper trading
        
        Args:
            decision: Decision dict with details
            
        Returns:
            Dict with simulated order execution
        """
        # Simulate a short delay for realism
        await asyncio.sleep(0.5)
        
        # Extract decision details
        decision_type = decision['decision']
        current_price = decision.get('current_price', 0)
        
        # Determine side from decision type
        if decision_type in ('ENTER_LONG', 'EXIT_SHORT'):
            side = 'BUY'
        else:
            side = 'SELL'
            
        # Simulate a fixed quantity
        quantity = 0.1  # Placeholder
        
        # Create simulated order
        simulated_order = {
            'orderId': f"sim_{int(time.time())}",
            'symbol': self.exchange_symbol,
            'side': side,
            'type': 'MARKET',
            'status': 'FILLED',
            'origQty': quantity,
            'executedQty': quantity,
            'avgPrice': current_price,
            'commission': self.config['FEE_PER_TRADE'] * quantity * current_price,
            'commissionAsset': 'USDT',
            'time': int(time.time() * 1000)
        }
        
        # Log decision and simulated execution to database
        decision_id = await self._log_decision(decision)
        if decision_id:
            await self.db_logger.log_executed_trade(decision_id, simulated_order)
        
        # Update position state in database based on decision
        position_state = 'NONE'
        entry_price = None
        position_size = None
        
        if decision_type == 'ENTER_LONG':
            position_state = 'LONG'
            entry_price = current_price
            position_size = quantity
            logger.info(f"Opening simulated LONG position: price={current_price}, size={quantity}")
        elif decision_type == 'ENTER_SHORT':
            position_state = 'SHORT'
            entry_price = current_price
            position_size = quantity
            logger.info(f"Opening simulated SHORT position: price={current_price}, size={quantity}")
        elif decision_type in ('EXIT_LONG', 'EXIT_SHORT'):
            logger.info(f"Closing simulated {'LONG' if decision_type == 'EXIT_LONG' else 'SHORT'} position at price {current_price}")
        
        # Update position state in database
        await self.db_logger.update_position_state(
            decision['trading_pair'],
            position_state,
            entry_price=entry_price,
            position_size=position_size,
            entry_timestamp=datetime.now() if position_state != 'NONE' else None
        )
        
        logger.info(f"Position state updated in database: {position_state}")
        logger.info(f"Simulated {side} order for {quantity} {self.trading_pair} at {current_price}")
        return simulated_order
    
    async def _log_decision(self, decision):
        """
        Log a trading decision to the database
        
        Args:
            decision: Decision dict
            
        Returns:
            Decision ID or None if error
        """
        try:
            decision_id = await self.db_logger.log_trade_decision(
                timestamp=decision['timestamp'],
                trading_pair=decision['trading_pair'],
                prediction_id=None,  # In a real system, this would be linked to a prediction
                decision=decision['decision'],
                target_price=decision.get('current_price'),
                prediction_value=decision.get('prediction_value'),
                reason=decision.get('reason')
            )
            return decision_id
            
        except Exception as e:
            logger.error(f"Error logging trade decision: {str(e)}")
            return None
    
    def _calculate_quantity(self, decision):
        """
        Calculate order quantity based on position sizing rules
        
        Args:
            decision: Decision dict
            
        Returns:
            Quantity to order
        """
        # In a real implementation, this would use account balance, position sizing
        # rules from the decision engine, etc.
        # For now, return a placeholder value
        return 0.1 