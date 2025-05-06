import logging
import asyncio
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderBookProcessor:
    """
    Processes order book updates from WebSocket and maintains order book state
    """
    
    def __init__(self, config, data_fetcher, db_logger):
        self.config = config
        self.data_fetcher = data_fetcher
        self.db_logger = db_logger
        self.trading_pair = config['TRADING_PAIR']
        self.exchange_symbol = config['EXCHANGE_SYMBOL']
        self.last_update_id = 0
        self.order_events_buffer = []
        self.buffer_flush_size = 50  # Number of events before flushing to DB
        self.last_buffer_flush = time.time()
        self.buffer_flush_interval = 5  # Seconds between buffer flushes
        self.running = False
        
    async def initialize(self):
        """Initialize order book with snapshot"""
        try:
            # Try to get initial snapshot from REST API first
            snapshot = await self.data_fetcher.get_initial_order_book(self.trading_pair)
            if snapshot:
                self.last_update_id = snapshot['lastUpdateId']
                logger.info(f"Initial order book snapshot processed, last update ID: {self.last_update_id}")
                return True
            
            # If REST API fails (418/429 errors), fallback to WebSocket-only mode
            logger.warning("Failed to get initial REST order book snapshot. Starting in WebSocket-only mode.")
            logger.warning("Order book will be built gradually from WebSocket updates. Some initial prediction cycles may be less accurate.")
            
            # Initialize with empty order book and a placeholder last_update_id
            # We'll rebuild it from WebSocket data
            self.last_update_id = 0
            return True
                
        except Exception as e:
            logger.error(f"Error during order book initialization: {str(e)}")
            return False
        
    async def start_processing(self):
        """Start processing order book updates from WebSocket"""
        # Mark as running
        self.running = True
        
        # Define stream name for Binance WebSocket
        depth_stream = f"{self.exchange_symbol.lower()}@depth@{self.config['DEPTH_STREAM_INTERVAL']}"
        logger.info(f"Starting order book processor for {self.trading_pair} (stream: {depth_stream})")
        
        # Start processing
        async for depth_update in self.data_fetcher.start_depth_stream(depth_stream):
            # Check if we should stop
            if not self.running:
                logger.info("Order book processor stop requested")
                break
                
            # Process the update
            await self._process_depth_update(depth_update)
            
            # Check if we should flush the buffer
            current_time = time.time()
            if (len(self.order_events_buffer) >= self.buffer_flush_size or 
                current_time - self.last_buffer_flush >= self.buffer_flush_interval):
                await self._flush_order_events_buffer()
    
    async def stop_processing(self):
        """Stop processing order book updates"""
        self.running = False
        logger.info("Order book processor stopping")
        
        # Flush any remaining events
        await self._flush_order_events_buffer()
    
    async def _process_depth_update(self, update):
        """
        Process a single depth update message
        
        Args:
            update: Dict with depth update data
        """
        try:
            # Special case for WebSocket-only initialization
            if self.last_update_id == 0:
                # Starting from scratch - just accept the first update without gap detection
                logger.info(f"Processing first WebSocket update in WebSocket-only mode. Update ID: {update['u']}")
                self.last_update_id = update['u']
                timestamp = datetime.fromtimestamp(update['E'] / 1000.0)
                
                # Process the initial update
                current_order_book = self.data_fetcher.get_current_order_book()
                
                # Process bids and asks just like normal updates
                if 'b' in update:
                    for bid_update in update['b']:
                        price, qty = bid_update
                        price_str = str(float(price))  # Normalize price format
                        qty_float = float(qty)
                        
                        # Update local order book
                        if qty_float == 0:
                            # Remove price level
                            if price_str in current_order_book['bids']:
                                del current_order_book['bids'][price_str]
                        else:
                            # Update price level
                            current_order_book['bids'][price_str] = {
                                'quantity': qty_float
                            }
                
                if 'a' in update:
                    for ask_update in update['a']:
                        price, qty = ask_update
                        price_str = str(float(price))  # Normalize price format
                        qty_float = float(qty)
                        
                        # Update local order book
                        if qty_float == 0:
                            # Remove price level
                            if price_str in current_order_book['asks']:
                                del current_order_book['asks'][price_str]
                        else:
                            # Update price level
                            current_order_book['asks'][price_str] = {
                                'quantity': qty_float
                            }
                
                return
                
            # Skip updates with earlier update IDs
            if 'u' not in update or update['u'] <= self.last_update_id:
                return
                
            # Extract order book data
            timestamp = datetime.fromtimestamp(update['E'] / 1000.0)
            first_update_id = update['U']
            final_update_id = update['u']
            
            # Check if we missed any updates
            if first_update_id > self.last_update_id + 1:
                gap_size = first_update_id - (self.last_update_id + 1)
                logger.warning(f"Gap in order book updates detected: {self.last_update_id} -> {first_update_id}")
                
                # If the gap is significant (more than 10000 updates missed), re-fetch the snapshot
                # This threshold is increased to avoid frequent REST API calls
                if gap_size > 10000:
                    logger.warning(f"Large update gap detected ({gap_size} updates). Adding short delay before attempting snapshot refresh.")
                    # Add a small delay to avoid hammering the API if gaps occur frequently
                    await asyncio.sleep(1.5) # Delay for 1.5 seconds
                    
                    logger.info(f"Attempting to refresh order book snapshot after gap...")
                    snapshot = await self.data_fetcher.get_initial_order_book(self.trading_pair)
                    
                    if snapshot:
                        self.last_update_id = snapshot['lastUpdateId']
                        logger.info(f"Order book snapshot refreshed. New last update ID: {self.last_update_id}")
                        
                        # Skip this update if it's now outdated
                        if final_update_id <= self.last_update_id:
                            return
                    else:
                        logger.error("Failed to refresh order book snapshot. Continuing with existing data.")
            
            # Update the current order book with this data
            current_order_book = self.data_fetcher.get_current_order_book()
            
            # Process bid updates (price decreases)
            if 'b' in update:
                for bid_update in update['b']:
                    price, qty = bid_update
                    price_str = str(float(price))  # Normalize price format
                    qty_float = float(qty)
                    
                    # Generate order events for tracking
                    event = self._create_order_event(price_str, qty_float, 'BID', timestamp)
                    if event:
                        self.order_events_buffer.append(event)
                    
                    # Update local order book
                    if qty_float == 0:
                        # Remove price level
                        if price_str in current_order_book['bids']:
                            del current_order_book['bids'][price_str]
                    else:
                        # Update price level
                        current_order_book['bids'][price_str] = {
                            'quantity': qty_float
                        }
            
            # Process ask updates (price increases)
            if 'a' in update:
                for ask_update in update['a']:
                    price, qty = ask_update
                    price_str = str(float(price))  # Normalize price format
                    qty_float = float(qty)
                    
                    # Generate order events for tracking
                    event = self._create_order_event(price_str, qty_float, 'ASK', timestamp)
                    if event:
                        self.order_events_buffer.append(event)
                    
                    # Update local order book
                    if qty_float == 0:
                        # Remove price level
                        if price_str in current_order_book['asks']:
                            del current_order_book['asks'][price_str]
                    else:
                        # Update price level
                        current_order_book['asks'][price_str] = {
                            'quantity': qty_float
                        }
            
            # Update last update ID
            self.last_update_id = final_update_id
            
        except Exception as e:
            logger.error(f"Error processing depth update: {type(e).__name__} - {str(e)}")
    
    def _create_order_event(self, price, quantity, side, timestamp):
        """
        Create an order event from an order book update
        
        Args:
            price: Price level
            quantity: New quantity
            side: 'BID' or 'ASK'
            timestamp: Event timestamp
            
        Returns:
            Order event dict or None if not applicable
        """
        # Generate a synthetic order ID based on price and side
        # In a real system, we would have real order IDs
        synthetic_order_id = f"{side}_{price}_{timestamp.timestamp()}"
        
        current_order_book = self.data_fetcher.get_current_order_book()
        price_levels = current_order_book['bids'] if side == 'BID' else current_order_book['asks']
        
        # Determine event type
        if price in price_levels:
            current_qty = price_levels[price]['quantity']
            if quantity == 0:
                # Order removed
                event_type = 'CANCEL'
                remaining_quantity = 0
            elif quantity != current_qty:
                # Order modified
                event_type = 'MODIFY'
                remaining_quantity = quantity
            else:
                # No change (shouldn't happen, but handle gracefully)
                return None
        else:
            if quantity > 0:
                # New order
                event_type = 'NEW'
                remaining_quantity = quantity
            else:
                # Zero quantity for non-existent price (shouldn't happen, but handle gracefully)
                return None
        
        # Create the event
        return {
            'trading_pair': self.trading_pair,
            'synthetic_order_id': synthetic_order_id,
            'event_type': event_type,
            'timestamp': timestamp,
            'price': float(price),
            'quantity': float(quantity),
            'side': side,
            'remaining_quantity': remaining_quantity
        }
    
    async def _flush_order_events_buffer(self):
        """Flush order events buffer to the database"""
        if not self.order_events_buffer:
            return
            
        events_to_flush = self.order_events_buffer.copy()
        self.order_events_buffer = []
        self.last_buffer_flush = time.time()
        
        # Log events to database
        events_logged = await self.db_logger.log_order_events(events_to_flush)
        logger.debug(f"Flushed {events_logged} order events to database") 