import logging
import json
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import websockets
from datetime import datetime, timedelta
import random  # Added for jitter
from collections import deque

logger = logging.getLogger(__name__)

# Define retry parameters
MAX_RETRIES = 4  # Max number of retries (e.g., 1s, 2s, 4s, 8s)
INITIAL_BACKOFF = 1  # Initial backoff delay in seconds
MAX_BACKOFF = 10 # Max backoff delay
# Add: Cooldown after 418 error
COOLDOWN_AFTER_418 = 120  # 2 minutes cooldown after a 418 error

class DataFetcher:
    """
    Handles data retrieval from Binance API for both OHLCV and order book data
    """
    
    def __init__(self, config):
        self.config = config
        self.rest_session = None
        self.last_update_id = 0
        self.current_order_book = {
            'bids': {},  # price -> {quantity, orders}
            'asks': {}   # price -> {quantity, orders}
        }
        # Set a User-Agent
        self.user_agent = f"RealtimeTFTBot/1.0 (Contact: your_email@example.com)" # Replace with actual contact if desired
        # Add: Last 418 error time tracking and cooldown
        self.last_418_error_time = 0
        self.rest_api_calls = []  # Track timestamps of API calls
        self.max_calls_per_minute = 5  # Conservative limit
        
        # Add: Cached OHLCV data
        self.cached_ohlcv = None
        self.last_ohlcv_update = 0
        
        # Add: WebSocket-based OHLCV storage
        self.ws_klines = {}  # symbol -> interval -> list of klines
        self.kline_buffer_size = max(200, config['CONTEXT_LENGTH'] * 3)  # Store 3x the required context
        self.kline_streams = set()  # Track active kline streams
        self.kline_initialized = {}  # Track which symbol+intervals have been initialized 
        
    async def initialize(self):
        """Initialize HTTP session for API calls"""
        # Include User-Agent in session headers
        headers = {
            'User-Agent': self.user_agent
        }
        self.rest_session = aiohttp.ClientSession(headers=headers)
        logger.info("Data fetcher initialized")
        return True
        
    async def close(self):
        """Close all connections"""
        if self.rest_session and not self.rest_session.closed:
            await self.rest_session.close()
            logger.info("REST session closed")
    
    async def get_latest_ohlcv(self, trading_pair, interval_seconds, limit=100):
        """
        Get OHLCV data for a trading pair - now with WebSocket support
        
        Args:
            trading_pair: Trading pair (e.g., 'WLD-USDT')
            interval_seconds: Interval in seconds (will be converted to Binance format)
            limit: Number of candles to retrieve
            
        Returns:
            Pandas DataFrame with OHLCV data
        """
        # Convert trading pair to Binance format
        symbol = trading_pair.replace('-', '')
        
        # Map interval_seconds to Binance interval format
        interval_map = {
            10: '1m',     # 10 seconds - use 1m as fallback for futures
            60: '1m',     # 1 minute
            180: '3m',    # 3 minutes
            300: '5m',    # 5 minutes
            900: '15m',   # 15 minutes
            1800: '30m',  # 30 minutes
            3600: '1h',   # 1 hour
            14400: '4h',  # 4 hours
            86400: '1d',  # 1 day
        }
        
        # Find the closest interval if exact match not found
        if interval_seconds not in interval_map:
            closest_key = min(interval_map.keys(), key=lambda x: abs(x - interval_seconds))
            interval = interval_map[closest_key]
            logger.warning(f"Interval {interval_seconds}s not directly supported. Using {closest_key}s ({interval}) instead.")
        else:
            interval = interval_map[interval_seconds]
        
        # Generate a unique key for this symbol+interval combination
        kline_key = f"{symbol}_{interval}"
        
        # Check if we need to initialize this kline stream
        if kline_key not in self.kline_initialized or not self.kline_initialized[kline_key]:
            try:
                # Start the WebSocket kline stream for this symbol and interval
                await self._initialize_kline_stream(symbol, interval)
                
                # Attempt to bootstrap with historical data (but handle failure gracefully)
                await self._bootstrap_historical_klines(symbol, interval)
                
                # Mark as initialized
                self.kline_initialized[kline_key] = True
                
            except Exception as e:
                logger.error(f"Error initializing kline stream for {kline_key}: {str(e)}")
                # If WebSocket setup fails, we'll try REST API as fallback
        
        # If we have cached WebSocket data for this symbol+interval, use that
        if kline_key in self.ws_klines and len(self.ws_klines[kline_key]) >= self.config['CONTEXT_LENGTH']:
            # Convert the stored klines to a DataFrame
            df = self._klines_to_dataframe(self.ws_klines[kline_key], limit)
            logger.info(f"Using WebSocket-based OHLCV data for {trading_pair}, interval {interval} ({len(df)} rows)")
            return df
            
        # If WebSocket data is insufficient, try using the REST API as fallback (this might fail with 418)
        if not self.rest_session:
            logger.error("REST session not initialized. Call initialize() first.")
            return None
        
        # Check if we're in a cooldown period after a 418 error
        current_time = time.time()
        if current_time - self.last_418_error_time < COOLDOWN_AFTER_418:
            remaining_cooldown = COOLDOWN_AFTER_418 - (current_time - self.last_418_error_time)
            logger.warning(f"In cooldown period after 418 error. {remaining_cooldown:.0f}s remaining before retry allowed.")
            
            # Return cached data if available as fallback
            if self.cached_ohlcv is not None:
                logger.info(f"Using cached OHLCV data ({len(self.cached_ohlcv)} rows) during API cooldown")
                return self.cached_ohlcv
                
            return None
        
        try:
            # For 10s interval, use spot market API as futures doesn't support it
            if interval_seconds == 10:
                # Use spot API for 1m data (as fallback)
                base_url = 'https://api.binance.com'
                url = f"{base_url}/api/v3/klines"
                logger.info(f"Using spot market API for {interval_seconds}s interval")
            else:
                # Use futures API for all other intervals
                base_url = self.config['BINANCE_BASE_URL']
                url = f"{base_url}/fapi/v1/klines"
            
            # Make the request
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            # Rate limiting check
            self.rest_api_calls = [t for t in self.rest_api_calls if current_time - t < 60]
            if len(self.rest_api_calls) >= self.max_calls_per_minute:
                wait_time = 60 - (current_time - self.rest_api_calls[0]) + 1
                logger.warning(f"Rate limiting: Waiting {wait_time:.2f}s before REST call")
                await asyncio.sleep(wait_time)
            
            # Track this call
            self.rest_api_calls.append(current_time)
            
            # Make the request
            async with self.rest_session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error fetching OHLCV data: HTTP {response.status}")
                    logger.error(f"Response: {error_text}")
                    
                    # Update last 418 error time if needed
                    if response.status == 418:
                        self.last_418_error_time = time.time()
                        
                    # Return cached data if available
                    if self.cached_ohlcv is not None:
                        logger.info(f"Using cached OHLCV data ({len(self.cached_ohlcv)} rows) after REST API error")
                        return self.cached_ohlcv
                        
                    return None
                
                data = await response.json()
            
            # Parse the response into a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Cache the result
            self.cached_ohlcv = df
            self.last_ohlcv_update = time.time()
            
            logger.info(f"Fetched {len(df)} OHLCV rows for {trading_pair}, interval {interval} via REST API")
            return df
            
        except Exception as e:
            logger.error(f"Error in get_latest_ohlcv: {type(e).__name__} - {str(e)}")
            
            # Return cached data if available
            if self.cached_ohlcv is not None:
                logger.info(f"Using cached OHLCV data ({len(self.cached_ohlcv)} rows) after exception")
                return self.cached_ohlcv
                
            return None
            
    async def _bootstrap_historical_klines(self, symbol, interval):
        """
        Bootstrap initial kline data from historical API for WebSocket
        Falls back gracefully if REST API is unavailable
        """
        # Only try if we're not in cooldown
        current_time = time.time()
        if current_time - self.last_418_error_time < COOLDOWN_AFTER_418:
            logger.warning(f"Skipping historical kline bootstrap during 418 cooldown period")
            return False
            
        try:
            # Rate limiting check
            self.rest_api_calls = [t for t in self.rest_api_calls if current_time - t < 60]
            if len(self.rest_api_calls) >= self.max_calls_per_minute:
                wait_time = 60 - (current_time - self.rest_api_calls[0]) + 1
                logger.warning(f"Rate limiting: Waiting {wait_time:.2f}s before bootstrap REST call")
                await asyncio.sleep(wait_time)
            
            # Track this call
            self.rest_api_calls.append(current_time)
            
            # Use futures API (assumed from context)
            base_url = self.config['BINANCE_BASE_URL']
            url = f"{base_url}/fapi/v1/klines"
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(100, self.kline_buffer_size)  # Only request what we need
            }
            
            async with self.rest_session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f"Failed to bootstrap historical klines: HTTP {response.status}. Response: {error_text}")
                    if response.status == 418:
                        self.last_418_error_time = time.time()
                    return False
                
                data = await response.json()
                
            # Initialize the klines storage if needed
            kline_key = f"{symbol}_{interval}"
            self.ws_klines[kline_key] = deque(maxlen=self.kline_buffer_size)
            
            # Process and store the klines
            for kline_data in data:
                kline = self._process_rest_kline(kline_data)
                self.ws_klines[kline_key].append(kline)
                
            logger.info(f"Bootstrapped {len(data)} historical {interval} klines for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error bootstrapping historical klines: {type(e).__name__} - {str(e)}")
            return False
    
    def _process_rest_kline(self, kline_data):
        """Convert REST API kline format to our internal format"""
        return {
            'event_time': int(kline_data[0]),  # open time
            'start_time': int(kline_data[0]),
            'close_time': int(kline_data[6]),
            'symbol': None,  # Will be filled from context
            'interval': None,  # Will be filled from context  
            'open': float(kline_data[1]),
            'close': float(kline_data[4]),
            'high': float(kline_data[2]),
            'low': float(kline_data[3]),
            'volume': float(kline_data[5]),
            'trades': int(kline_data[8]),
            'complete': True  # Historical candles are complete
        }
    
    def _process_ws_kline(self, kline_msg):
        """Extract kline from WebSocket message"""
        k = kline_msg['k']
        return {
            'event_time': kline_msg['E'],
            'start_time': k['t'],
            'close_time': k['T'],
            'symbol': k['s'],
            'interval': k['i'],
            'open': float(k['o']),
            'close': float(k['c']),
            'high': float(k['h']),
            'low': float(k['l']),
            'volume': float(k['v']),
            'trades': int(k['n']),
            'complete': k['x']  # Whether this candle is complete
        }
    
    def _klines_to_dataframe(self, klines, limit):
        """Convert internal kline format to pandas DataFrame"""
        # Convert to list and get the most recent 'limit' klines
        kline_list = list(klines)[-limit:]
        
        # Extract the data points
        data = []
        for kline in kline_list:
            data.append([
                kline['start_time'],
                kline['open'],
                kline['high'], 
                kline['low'],
                kline['close'],
                kline['volume'],
                kline['close_time'],
                0.0,  # quote_volume (not used)
                kline['trades'],
                0.0,  # taker_buy_base (not used)
                0.0,  # taker_buy_quote (not used)
                0    # ignored (not used)
            ])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    async def _initialize_kline_stream(self, symbol, interval):
        """Start the WebSocket kline stream for a specific symbol and interval"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        
        # If already started, don't start again
        if stream_name in self.kline_streams:
            return
            
        # Add to set of streams
        self.kline_streams.add(stream_name)
        
        # Start the stream in the background
        asyncio.create_task(self._kline_stream_handler(stream_name, symbol, interval))
        logger.info(f"Started kline stream for {symbol} {interval}")
    
    async def _kline_stream_handler(self, stream_name, symbol, interval):
        """Handle the kline WebSocket stream"""
        binance_ws_url = "wss://fstream.binance.com/ws"
        kline_key = f"{symbol}_{interval}"
        
        # Initialize storage if not exists
        if kline_key not in self.ws_klines:
            self.ws_klines[kline_key] = deque(maxlen=self.kline_buffer_size)
        
        while True:
            try:
                logger.info(f"Connecting to kline stream {stream_name}...")
                
                async with websockets.connect(binance_ws_url) as websocket:
                    # Subscribe to the stream
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [stream_name],
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    # Check subscription response
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if 'result' in response_data and response_data['result'] is None:
                        logger.info(f"Successfully subscribed to kline stream {stream_name}")
                    else:
                        logger.error(f"Failed to subscribe to {stream_name}: {response_data}")
                        await asyncio.sleep(5)
                        continue
                    
                    # Process messages
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # Skip subscription responses
                            if 'result' in data:
                                continue
                                
                            # Ensure it's a kline message
                            if 'e' in data and data['e'] == 'kline':
                                kline = self._process_ws_kline(data)
                                
                                # Update existing kline or add new one
                                updated = False
                                
                                # If this is an update to a kline we already have, update it
                                for i, existing_kline in enumerate(self.ws_klines[kline_key]):
                                    if existing_kline['start_time'] == kline['start_time']:
                                        self.ws_klines[kline_key][i] = kline
                                        updated = True
                                        break
                                
                                # If it's a new kline, add it
                                if not updated:
                                    self.ws_klines[kline_key].append(kline)
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"Kline WebSocket connection closed for {stream_name}")
                            break
                            
            except Exception as e:
                logger.error(f"Kline WebSocket error: {type(e).__name__} - {str(e)}")
                
            # Wait before reconnecting
            logger.info(f"Reconnecting kline stream {stream_name} in 5 seconds...")
            await asyncio.sleep(5)
    
    async def get_account_balance(self):
        """
        Get account balance (when API keys are provided)
        
        Returns:
            Dict with asset balances
        """
        if not self.config['BINANCE_API_KEY'] or not self.config['BINANCE_API_SECRET']:
            logger.error("API credentials missing, cannot fetch account balance")
            return None
            
        if not self.rest_session:
            logger.error("REST session not initialized. Call initialize() first.")
            return None
            
        try:
            # For futures trading, we need to use a different endpoint
            url = f"{self.config['BINANCE_BASE_URL']}/fapi/v2/account"
            
            # For authenticated requests, we need to sign the request
            # This would normally involve creating a signature with the API secret
            # But for now, let's just indicate this is where we'd do that
            # In a real implementation, we would use python-binance or a similar library
            
            headers = {
                'X-MBX-APIKEY': self.config['BINANCE_API_KEY']
            }
            
            # In a real implementation, we would add the signature parameters
            # For now, just log a placeholder message
            logger.info("In a production environment, the request would be signed with the API secret")
            
            # This won't work without proper authentication
            # Commented out to avoid errors
            """
            async with self.rest_session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Error fetching account data: HTTP {response.status}")
                    return None
                
                data = await response.json()
            """
            
            # Return placeholder data
            return {"message": "Account balance would be fetched here in production"}
            
        except Exception as e:
            logger.error(f"Error fetching account balance: {str(e)}")
            return None
    
    async def get_initial_order_book(self, trading_pair, depth=100):
        """
        Get initial order book snapshot from REST API with retry logic
        
        Args:
            trading_pair: Trading pair (e.g., 'WLD-USDT')
            depth: Order book depth to retrieve (default 100)
            
        Returns:
            Dict with order book data or None if fails after retries
        """
        if not self.rest_session:
            logger.error("REST session not initialized. Call initialize() first.")
            return None
        
        # Add: Check if we're in a cooldown period after a 418 error
        current_time = time.time()
        if current_time - self.last_418_error_time < COOLDOWN_AFTER_418:
            remaining_cooldown = COOLDOWN_AFTER_418 - (current_time - self.last_418_error_time)
            logger.warning(f"In cooldown period after 418 error. {remaining_cooldown:.0f}s remaining before retry allowed.")
            return None
        
        # Add: Rate limiting check
        # Clean up old calls (older than 60 seconds)
        self.rest_api_calls = [t for t in self.rest_api_calls if current_time - t < 60]
        
        # Check if we're over limit
        if len(self.rest_api_calls) >= self.max_calls_per_minute:
            wait_time = 60 - (current_time - self.rest_api_calls[0]) + 1
            logger.warning(f"Rate limiting: Waiting {wait_time:.2f}s before REST call")
            await asyncio.sleep(wait_time)
        
        # Track this call
        self.rest_api_calls.append(current_time)
            
        try:
            # Convert trading pair to Binance format
            symbol = trading_pair.replace('-', '')
            
            # Binance API endpoint for order book
            url = f"{self.config['BINANCE_BASE_URL']}/fapi/v1/depth"
            params = {
                'symbol': symbol,
                'limit': depth
            }
            
            retries = 0
            backoff = INITIAL_BACKOFF
            
            while retries <= MAX_RETRIES:
                try:
                    # Make the request
                    async with self.rest_session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Update the local order book
                            self.last_update_id = data['lastUpdateId']
                            
                            # Process bids
                            self.current_order_book['bids'] = {}
                            for price_qty in data.get('bids', []):
                                price, qty = float(price_qty[0]), float(price_qty[1])
                                if qty > 0:  # Only store non-zero quantities
                                    price_str = str(price)
                                    self.current_order_book['bids'][price_str] = {
                                        'quantity': qty
                                    }
                            
                            # Process asks
                            self.current_order_book['asks'] = {}
                            for price_qty in data.get('asks', []):
                                price, qty = float(price_qty[0]), float(price_qty[1])
                                if qty > 0:  # Only store non-zero quantities
                                    price_str = str(price)
                                    self.current_order_book['asks'][price_str] = {
                                        'quantity': qty
                                    }
                            
                            logger.info(f"Initial order book fetched with {len(self.current_order_book['bids'])} bids and {len(self.current_order_book['asks'])} asks")
                            return data # Success
                        elif response.status in [418, 429]: # Rate limit or banned
                            logger.warning(f"Rate limited (HTTP {response.status}) fetching order book. Retry {retries + 1}/{MAX_RETRIES} in {backoff:.2f}s...")
                            # Add: Set the last 418 error time
                            if response.status == 418:
                                self.last_418_error_time = time.time()
                            # Fall through to retry logic below
                        else:
                            # Other non-200 error, log and give up immediately
                            error_text = await response.text()
                            logger.error(f"Error fetching order book: HTTP {response.status}. Response: {error_text}")
                            return None # Don't retry on unexpected errors
                            
                except aiohttp.ClientError as e:
                    logger.warning(f"Network error fetching order book: {type(e).__name__}. Retry {retries + 1}/{MAX_RETRIES} in {backoff:.2f}s...")
                    # Fall through to retry logic below
                except Exception as e:
                    # Catch unexpected errors during processing
                    logger.error(f"Unexpected error fetching/processing order book: {type(e).__name__} - {str(e)}")
                    return None # Don't retry on unexpected logic errors
                
                # If we reached here, it means we need to retry (due to 418, 429, or network error)
                retries += 1
                if retries <= MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2 + random.uniform(0, 0.1 * backoff), MAX_BACKOFF) # Exponential backoff with jitter
                else:
                    logger.error(f"Failed to fetch order book after {MAX_RETRIES} retries.")
            
            return None # Failed after all retries
            
        except Exception as e:
            logger.error(f"Error fetching initial order book: {type(e).__name__} - {str(e)}")
            return None
    
    def get_current_order_book(self):
        """Get the current order book state"""
        return self.current_order_book
    
    async def start_depth_stream(self, stream_name):
        """
        Connect to WebSocket and yield depth update messages
        
        Args:
            stream_name: Name of the WebSocket stream to connect to
            
        Yields:
            Dict with depth update data
        """
        binance_ws_url = "wss://fstream.binance.com/ws"
        
        while True:
            try:
                logger.info(f"Connecting to {stream_name}...")
                
                async with websockets.connect(binance_ws_url) as websocket:
                    # Subscribe to the stream
                    subscribe_msg = {
                        "method": "SUBSCRIBE",
                        "params": [stream_name],
                        "id": 1
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    # Check subscription response
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    if 'result' in response_data and response_data['result'] is None:
                        logger.info(f"Successfully subscribed to {stream_name}")
                    else:
                        logger.error(f"Failed to subscribe to {stream_name}: {response_data}")
                        await asyncio.sleep(5)
                        continue
                    
                    # Process messages
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            # Skip subscription responses
                            if 'result' in data:
                                continue
                                
                            # Yield the message to be processed
                            yield data
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed")
                            break
                            
            except Exception as e:
                logger.error(f"WebSocket error: {type(e).__name__} - {str(e)}")
                
            # Wait before reconnecting
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            
    async def get_tickers(self, symbol=None):
        """
        Get current ticker information (latest prices)
        
        Args:
            symbol: Optional specific symbol to get data for
            
        Returns:
            List of ticker data or single ticker dict
        """
        if not self.rest_session:
            logger.error("REST session not initialized. Call initialize() first.")
            return None
            
        try:
            # Binance API endpoint for tickers
            url = f"{self.config['BINANCE_BASE_URL']}/fapi/v1/ticker/24hr"
            params = {}
            
            if symbol:
                params['symbol'] = symbol
                
            # Make the request
            async with self.rest_session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error fetching ticker data: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
            logger.debug(f"Fetched ticker data {'for ' + symbol if symbol else 'for all symbols'}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching ticker data: {str(e)}")
            return None
    
    async def get_symbol_info(self, symbol=None):
        """
        Get exchange information about trading symbols
        
        Args:
            symbol: Optional specific symbol to get data for
            
        Returns:
            Dict with exchange information
        """
        if not self.rest_session:
            logger.error("REST session not initialized. Call initialize() first.")
            return None
            
        try:
            # Binance API endpoint for exchange info
            url = f"{self.config['BINANCE_BASE_URL']}/fapi/v1/exchangeInfo"
            params = {}
            
            if symbol:
                params['symbol'] = symbol
                
            # Make the request
            async with self.rest_session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error fetching exchange info: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
            logger.debug(f"Fetched exchange info {'for ' + symbol if symbol else 'for all symbols'}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching exchange info: {str(e)}")
            return None 