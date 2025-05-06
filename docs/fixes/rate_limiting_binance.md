# Binance API Rate Limiting Fix

## Issue Description

### Problem Summary
The TFT bot experiences frequent HTTP 418 errors from Binance API, leading to IP bans and data gaps. This occurs because:

1. The order book processor requests new snapshots too frequently when gaps are detected
2. No exponential backoff is implemented for API retries
3. No circuit breaker exists to prevent overwhelming the API during rate limit periods
4. Depth stream interval is set too low (100ms)

### Logs
```
2025-05-03 22:36:23,222 - utils.data_fetcher - ERROR - Error fetching OHLCV data: HTTP 418
2025-05-03 22:36:23,223 - utils.data_fetcher - ERROR - Response: {"code":-1003,"msg":"Way too many requests; IP(194.127.199.97) banned until 1746327239762. Please use the websocket for live updates to avoid bans."}
```

### Impact
- Inability to fetch complete order book data
- Missing OHLCV data for model predictions
- Degraded prediction quality due to incomplete data
- Trading decisions based on partial or stale data
- IP bans preventing system from functioning

## Root Cause Analysis

### Problems in Current Implementation

#### 1. DataFetcher Class
- No rate limiting mechanism
- No exponential backoff for retries
- No tracking of request frequency
- No circuit breaker to prevent excessive requests

```python
# Current problematic implementation in data_fetcher.py
async def get_initial_order_book(self, symbol):
    """Fetch initial order book snapshot"""
    try:
        # This makes a direct request without rate limiting
        response = await self.session.get(f"{self.config['BINANCE_BASE_URL']}/api/v3/depth", 
                                         params={"symbol": symbol, "limit": 1000})
        if response.status == 200:
            data = await response.json()
            return data
        else:
            logger.error(f"Error fetching order book: HTTP {response.status}")
            return None
    except Exception as e:
        logger.error(f"Error fetching order book: {type(e).__name__} - {str(e)}")
        return None
```

#### 2. OrderBookProcessor Class
- Requests new snapshots too aggressively
- Gap detection thresholds too sensitive
- No backoff mechanism for snapshot refreshes
- No handling for long-term rate limit bans

```python
# Current problematic implementation in order_book_processor.py
async def _process_depth_update(self, update):
    # Check if we missed any updates
    if first_update_id > self.last_update_id + 1:
        gap_size = first_update_id - (self.last_update_id + 1)
        logger.warning(f"Gap in order book updates detected: {self.last_update_id} -> {first_update_id}")
        
        # Only refresh if the gap is very large and we haven't recently refreshed
        current_time = time.time()
        if gap_size > self.max_gap_tolerance and (current_time - self.last_snapshot_time) > self.min_snapshot_interval:
            logger.warning(f"Large update gap detected ({gap_size} updates). Re-fetching order book snapshot.")
            return False  # Signal that we need to refresh the snapshot
```

#### 3. Configuration Issues
- Depth stream interval too frequent (100ms)
- Insufficient time between snapshot refreshes (60 seconds)
- Gap tolerance threshold may be too low

## Solution Approach

### 1. Implement Rate Limiting and Backoff

#### DataFetcher Improvements

Add a rate limiter class to track and control API requests:

```python
class RateLimiter:
    def __init__(self):
        self.request_timestamps = defaultdict(list)
        self.rate_limits = {
            "order_book": {"limit": 10, "window": 60},  # 10 requests per minute
            "ohlcv": {"limit": 30, "window": 60},       # 30 requests per minute
            "default": {"limit": 6, "window": 60}       # 6 requests per minute for other endpoints
        }
        self.backoff_times = {}
        self.circuit_open = False
        self.circuit_reset_time = 0
        self.circuit_threshold = 5  # Number of failures before opening circuit
        self.circuit_reset_delay = 300  # 5 minutes
        self.failure_count = 0
        
    async def wait_if_needed(self, endpoint_type="default"):
        """Wait if we're approaching rate limits"""
        # Circuit breaker check
        if self.circuit_open:
            current_time = time.time()
            if current_time < self.circuit_reset_time:
                wait_time = self.circuit_reset_time - current_time
                logger.warning(f"Circuit breaker open. Waiting {wait_time:.2f}s before retry.")
                await asyncio.sleep(wait_time)
                self.circuit_open = False
                self.failure_count = 0
        
        # Apply endpoint-specific rate limits
        limit_config = self.rate_limits.get(endpoint_type, self.rate_limits["default"])
        limit = limit_config["limit"]
        window = limit_config["window"]
        
        # Clean old timestamps
        current_time = time.time()
        self.request_timestamps[endpoint_type] = [
            ts for ts in self.request_timestamps[endpoint_type] 
            if current_time - ts < window
        ]
        
        # Check if we need to wait
        if len(self.request_timestamps[endpoint_type]) >= limit * 0.8:  # 80% of limit
            # Calculate wait time based on oldest request
            if self.request_timestamps[endpoint_type]:
                oldest = min(self.request_timestamps[endpoint_type])
                wait_time = max(0, window - (current_time - oldest))
                if wait_time > 0:
                    logger.info(f"Rate limit approaching for {endpoint_type}. Waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_timestamps[endpoint_type].append(time.time())
    
    def record_failure(self, status_code):
        """Record API failure and possibly open circuit breaker"""
        if status_code == 418 or status_code == 429:
            self.failure_count += 1
            if self.failure_count >= self.circuit_threshold:
                self.circuit_open = True
                self.circuit_reset_time = time.time() + self.circuit_reset_delay
                logger.warning(f"Circuit breaker opened until {datetime.fromtimestamp(self.circuit_reset_time)}")
    
    def get_backoff_time(self, endpoint_type="default"):
        """Get exponential backoff time for retries"""
        if endpoint_type not in self.backoff_times:
            self.backoff_times[endpoint_type] = 1
            return 1
        
        # Exponential backoff with jitter
        self.backoff_times[endpoint_type] = min(60, self.backoff_times[endpoint_type] * 2)
        jitter = random.uniform(0.8, 1.2)
        return self.backoff_times[endpoint_type] * jitter
    
    def reset_backoff(self, endpoint_type="default"):
        """Reset backoff time after successful request"""
        self.backoff_times[endpoint_type] = 1
```

### 2. Modify OrderBookProcessor Gap Handling

Improve gap handling to reduce snapshot requests:

```python
async def _process_depth_update(self, update):
    # Check if we missed any updates
    if first_update_id > self.last_update_id + 1:
        gap_size = first_update_id - (self.last_update_id + 1)
        logger.warning(f"Gap in order book updates detected: {self.last_update_id} -> {first_update_id}")
        
        # More intelligent gap handling
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
        
        # For large gaps or when refresh is needed
        current_time = time.time()
        if (current_time - self.last_snapshot_time) > self.min_snapshot_interval:
            logger.warning(f"Large update gap detected ({gap_size} updates). Re-fetching order book snapshot.")
            return False  # Signal that we need to refresh the snapshot
        else:
            # If we refreshed recently, wait longer before another refresh
            logger.info(f"Gap detected but refreshed recently. Continuing with existing data.")
            self.last_update_id = first_update_id - 1
            return True
```

### 3. Update Configuration Parameters

```python
# Updated configuration values
config = {
    # ... existing config values
    'DEPTH_STREAM_INTERVAL': os.getenv('DEPTH_STREAM_INTERVAL', '1000ms'),  # Changed from 100ms to 1000ms
    'SMALL_GAP_TOLERANCE': int(os.getenv('SMALL_GAP_TOLERANCE', 1000)),  # Gaps smaller than this are ignored
    'MEDIUM_GAP_TOLERANCE': int(os.getenv('MEDIUM_GAP_TOLERANCE', 5000)),  # Medium gaps have different handling
    'MIN_SNAPSHOT_INTERVAL': int(os.getenv('MIN_SNAPSHOT_INTERVAL', 300)),  # 5 minutes between snapshots
    'MEDIUM_REFRESH_INTERVAL': int(os.getenv('MEDIUM_REFRESH_INTERVAL', 120)),  # 2 minutes for medium gaps
    'CIRCUIT_BREAKER_THRESHOLD': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', 5)),
    'CIRCUIT_BREAKER_DELAY': int(os.getenv('CIRCUIT_BREAKER_DELAY', 300)),  # 5 minutes
}
```

### 4. WebSocket Migration

Move from REST API to WebSocket for OHLCV data:

```python
async def start_kline_stream(self):
    """Start WebSocket stream for OHLCV data instead of REST API calls"""
    kline_stream = f"{self.config['EXCHANGE_SYMBOL'].lower()}@kline_{self.timeframe}"
    
    while self.running:
        try:
            logger.info(f"Starting kline WebSocket stream: {kline_stream}")
            async with self.session.ws_connect(f"{self.ws_base_url}/ws/{kline_stream}") as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        # Process kline data
                        if 'k' in data:
                            kline = data['k']
                            kline_data = {
                                'open_time': kline['t'] / 1000,
                                'open': float(kline['o']),
                                'high': float(kline['h']),
                                'low': float(kline['l']),
                                'close': float(kline['c']),
                                'volume': float(kline['v']),
                                'close_time': kline['T'] / 1000,
                                'trades': kline['n']
                            }
                            
                            # Update OHLCV data
                            await self._update_ohlcv_data(kline_data)
                    
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket connection closed")
                        break
                    
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        break
        
        except Exception as e:
            logger.error(f"Error in kline stream: {type(e).__name__} - {str(e)}")
            await asyncio.sleep(5)  # Wait before reconnecting
```

## Implementation Plan

### Phase 1: Emergency Fixes
1. Implement RateLimiter class in `utils/rate_limiter.py`
2. Integrate rate limiter with DataFetcher
3. Update OrderBookProcessor gap handling
4. Update configuration parameters

### Phase 2: WebSocket Migration
1. Implement WebSocket-based OHLCV data collection
2. Test and validate data quality
3. Phase out REST API calls for repetitive data

### Phase 3: Monitoring and Optimization
1. Add detailed metrics for API calls and rate limits
2. Fine-tune parameters based on production performance
3. Implement adaptive parameters that adjust based on market conditions

## Verification Steps

After implementation, verify the fix with these tests:

1. **Rate Limiting Test**
   - Run the bot for at least 30 minutes
   - Confirm no HTTP 418 errors in logs
   - Verify the circuit breaker functionality by simulating errors

2. **Gap Handling Test**
   - Analyze logs for gap detection events
   - Confirm that small gaps don't trigger snapshots
   - Verify proper handling of medium and large gaps

3. **WebSocket Stability Test**
   - Run the bot for 24 hours
   - Monitor WebSocket connection stability
   - Verify continuous data flow without interruptions 