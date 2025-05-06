# Rate Limiting & Order Book Processing Implementation

## Overview

This document describes the changes implemented to fix the HTTP 418 errors and IP bans encountered with the Binance API. The main issues were:

1. Excessive API requests due to aggressive order book snapshot refreshes
2. Lack of proper rate limiting and backoff
3. No circuit breaker to stop requests during rate limit periods
4. Inefficient depth stream interval (100ms)

## Implementation Details

### 1. RateLimiter Class (`utils/rate_limiter.py`)

Created a new `RateLimiter` class that provides:

- Request frequency tracking by endpoint type
- Waiting functionality before making requests that might exceed limits
- Exponential backoff for handling rate limit failures
- Circuit breaker pattern to prevent overwhelming the API

Key features:
```python
# Track request history by endpoint
self.request_timestamps = defaultdict(list)

# Define rate limits for different endpoints
self.rate_limits = {
    "order_book": {"limit": 10, "window": 60},  # 10 requests per minute
    "ohlcv": {"limit": 30, "window": 60},       # 30 requests per minute
    "default": {"limit": 6, "window": 60}       # 6 requests per minute for other endpoints
}

# Circuit breaker pattern
async def wait_if_needed(self, endpoint_type="default"):
    # Circuit breaker check
    if self.circuit_open:
        current_time = time.time()
        if current_time < self.circuit_reset_time:
            wait_time = self.circuit_reset_time - current_time
            logger.warning(f"Circuit breaker open. Waiting {wait_time:.2f}s before retry.")
            await asyncio.sleep(wait_time)
            self.circuit_open = False
            self.failure_count = 0
            
    # Rate limit checking and waiting logic...
```

### 2. Configuration Updates (`config.py`)

Added new configuration parameters with sensible defaults:

```python
# Rate Limiting & Order Book Processing
'DEPTH_STREAM_INTERVAL': os.getenv('DEPTH_STREAM_INTERVAL', '1000ms'),  # Changed from 100ms
'SMALL_GAP_TOLERANCE': int(os.getenv('SMALL_GAP_TOLERANCE', 1000)),     # Ignore small gaps
'MEDIUM_GAP_TOLERANCE': int(os.getenv('MEDIUM_GAP_TOLERANCE', 5000)),   # Different handling for medium gaps
'MIN_SNAPSHOT_INTERVAL': int(os.getenv('MIN_SNAPSHOT_INTERVAL', 300)),  # 5 minutes between snapshots
'MEDIUM_REFRESH_INTERVAL': int(os.getenv('MEDIUM_REFRESH_INTERVAL', 120)), # 2 minutes for medium gaps
'CIRCUIT_BREAKER_THRESHOLD': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', 5)),
'CIRCUIT_BREAKER_DELAY': int(os.getenv('CIRCUIT_BREAKER_DELAY', 300)),  # 5 minutes
'USE_WEBSOCKET_FOR_KLINES': os.getenv('USE_WEBSOCKET_FOR_KLINES', 'true').lower() == 'true',
'MAX_RETRIES': int(os.getenv('MAX_RETRIES', 3)),
```

### 3. DataFetcher Improvements (`utils/data_fetcher.py`)

Modified the DataFetcher class to:

- Use the new RateLimiter for all API calls
- Implement proper exponential backoff for retries
- Add WebSocket-based data collection for OHLCV data
- Handle reconnection for WebSocket streams

Key changes:
```python
async def get_initial_order_book(self, symbol):
    """Fetch initial order book snapshot with rate limiting and backoff"""
    max_retries = self.config.get('MAX_RETRIES', 3)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Wait if needed based on rate limits
            await self.rate_limiter.wait_if_needed("order_book")
            
            # Make request...
            
            if response.status == 200:
                # Process successful response...
                self.rate_limiter.reset_backoff("order_book")
                return data
            else:
                # Record failure and handle status code
                self.rate_limiter.record_failure(response.status)
                
                if response.status == 418 or response.status == 429:
                    # Apply exponential backoff for rate limiting errors
                    backoff_time = self.rate_limiter.get_backoff_time("order_book")
                    logger.warning(f"Rate limit hit. Backing off for {backoff_time:.2f}s before retry")
                    await asyncio.sleep(backoff_time)
                    retry_count += 1
```

Added WebSocket for OHLCV data:
```python
async def start_kline_stream(self):
    """Start WebSocket stream for OHLCV data instead of REST API calls"""
    kline_stream = f"{self.symbol.lower()}@kline_{self.timeframe}"
    reconnect_delay = 5  # seconds
    
    while self.running:
        try:
            # Connect to WebSocket and process messages...
        except Exception as e:
            # Handle error and implement reconnection with increasing delay
            reconnect_delay = min(30, reconnect_delay * 1.5)
```

### 4. OrderBookProcessor Improvements (`utils/order_book_processor.py`)

Modified the OrderBookProcessor to implement smarter gap handling:

```python
async def _process_depth_update(self, update):
    # ...
    
    # Check if we missed any updates
    if first_update_id > self.last_update_id + 1:
        gap_size = first_update_id - (self.last_update_id + 1)
        logger.warning(f"Gap in order book updates detected: {self.last_update_id} -> {first_update_id}")
        
        # More intelligent gap handling based on gap size
        if gap_size <= self.small_gap_tolerance:
            # For small gaps, continue without snapshot
            logger.info(f"Small gap of {gap_size} updates, continuing without snapshot")
            self.last_update_id = first_update_id - 1  # Set last ID to just before this update
            return True
        elif gap_size <= self.medium_gap_tolerance:
            # For medium gaps, only refresh if we haven't recently
            current_time = time.time()
            refresh_needed = (current_time - self.last_snapshot_time) > self.medium_refresh_interval
            if not refresh_needed:
                logger.info(f"Medium gap of {gap_size} updates, continuing with existing data")
                self.last_update_id = first_update_id - 1
                return True
```

## Verification

To verify the changes:

1. **Rate limiting logs**: The logs should show waiting and backoff when approaching rate limits
2. **No HTTP 418 errors**: The system should no longer receive HTTP 418 errors
3. **Fewer snapshot requests**: The system should request significantly fewer order book snapshots
4. **WebSocket stability**: The WebSocket connections should remain stable and reconnect if disconnected

Example expected log output:
```
2025-05-04 16:32:45,123 - utils.rate_limiter - INFO - Rate limit approaching for order_book. Waiting 3.45s
2025-05-04 16:32:45,123 - utils.order_book_processor - INFO - Small gap of 876 updates, continuing without snapshot
2025-05-04 16:32:45,123 - utils.order_book_processor - INFO - Medium gap of 3452 updates, continuing with existing data
```

## Configuration Recommendations

Based on testing, we recommend the following configuration values:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| DEPTH_STREAM_INTERVAL | 1000ms | Reduces update frequency to avoid overwhelming processing |
| SMALL_GAP_TOLERANCE | 1000 | Gaps under 1000 updates are usually recoverable |
| MEDIUM_GAP_TOLERANCE | 5000 | Gaps between 1000-5000 use more careful handling |
| MIN_SNAPSHOT_INTERVAL | 300 | At most 1 snapshot every 5 minutes |
| CIRCUIT_BREAKER_THRESHOLD | 5 | Open circuit after 5 consecutive failures |
| CIRCUIT_BREAKER_DELAY | 300 | Stay open for 5 minutes after triggering |

## Future Improvements

1. **Adaptive Parameters**: Implement dynamic adjustment of gap tolerance based on market volatility
2. **Enhanced Monitoring**: Add detailed metrics on API usage, gaps, and snapshot frequency
3. **Load Balancing**: Implement IP rotation for high-availability scenarios
4. **Further Optimization**: Analyze and optimize the order book update processing for efficiency 