from typing import List, Optional, Dict
import pandas as pd
import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator
from dataclasses import dataclass
from decimal import Decimal
import random
from datetime import datetime
import aiohttp
import json
import numpy as np


class EliteSmugPlugObiConfig(DirectionalTradingControllerConfigBase):
    controller_name = "elitesmugplugobi"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    ema_fast: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the fast EMA period (3-8): ",
            prompt_on_new=True))
    ema_medium: int = Field(
        default=12,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the medium EMA period (8-21): ",
            prompt_on_new=True))
    ema_slow: int = Field(
        default=13,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the slow EMA period (13-34): ",
            prompt_on_new=True))
    atr_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR length (10-21): ",
            prompt_on_new=True))
    atr_multiplier: float = Field(
        default=1.2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR multiplier (0.8-1.5): ",
            prompt_on_new=True))
    volume_surge: float = Field(
        default=1.6,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume surge multiplier (3.0-5.0): ",
            prompt_on_new=True))
    rsi_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI period (8-21): ",
            prompt_on_new=True))
    obi_window: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the OBI window length (10-30): ",
            prompt_on_new=True))
    obi_threshold: float = Field(
        default=0.6,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the OBI threshold (0.5-0.8): ",
            prompt_on_new=True))
    bid_ask_ratio_threshold: float = Field(
        default=1.2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the bid/ask ratio threshold (1.1-1.5): ",
            prompt_on_new=True))

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


@dataclass
class OrderBookSnapshot:
    """Order book snapshot structure"""
    bids: List[Dict[str, Decimal]]  # List of {price, amount} dictionaries
    asks: List[Dict[str, Decimal]]
    timestamp: int


class EliteSmugPlugObiController(DirectionalTradingControllerBase):
    """
    SmugPlug Emerald Elite v3 Lite
    High-Precision Breakout Detection with Multi-Timeframe Confirmation
    A lightweight version optimized for performance
    """

    def __init__(self, config: EliteSmugPlugObiConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def decimate(self, series):
        """Reduce signal noise by identifying unique signal changes."""
        start_of_sequence = series & ~series.shift(fill_value=False)
        group_number = start_of_sequence.cumsum()
        return series & (group_number == group_number.where(series).groupby(
            group_number).transform('first'))

    def generate_synthetic_orderbook(self, df: pd.DataFrame, depth: int = 10, spread_factor: float = 0.001) -> OrderBookSnapshot:
        """
        Generate synthetic order book data from OHLCV data.
        Uses price and volume to create realistic-looking order book snapshots.
        """
        current_close = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Calculate base spread around current price
        spread = current_close * spread_factor
        
        # Generate bid prices slightly below current price
        bid_prices = [current_close - (i * spread) for i in range(depth)]
        # Generate ask prices slightly above current price
        ask_prices = [current_close + (i * spread) for i in range(depth)]
        
        # Generate amounts based on current volume
        base_amount = current_volume / (depth * 2)  # Distribute volume across levels
        amounts = [base_amount * (0.8 + (0.4 * random.random())) for _ in range(depth)]
        
        # Create bid and ask entries
        bids = [{"price": Decimal(str(p)), "amount": Decimal(str(a))} 
               for p, a in zip(bid_prices, amounts)]
        asks = [{"price": Decimal(str(p)), "amount": Decimal(str(a))} 
               for p, a in zip(ask_prices, amounts)]
        
        return OrderBookSnapshot(bids=bids, asks=asks, timestamp=int(datetime.now().timestamp()))

    async def fetch_historical_orderbook(self, timestamp: int, limit: int = 50) -> Optional[OrderBookSnapshot]:
        """
        Fetch historical order book data from Binance API
        Documentation: https://binance-docs.github.io/apidocs/futures/en/#old-trade-lookup-market_data
        """
        # Convert timestamp to milliseconds if needed
        ts_ms = timestamp * 1000 if timestamp < 10000000000 else timestamp
        
        url = f"https://fapi.binance.com/fapi/v1/depth/snapshot"
        params = {
            "symbol": self.config.trading_pair.replace("-", ""),
            "limit": limit,
            "timestamp": ts_ms
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert string values to Decimal
                        bids = [{"price": Decimal(str(price)), "amount": Decimal(str(amount))} 
                               for price, amount in data["bids"]]
                        asks = [{"price": Decimal(str(price)), "amount": Decimal(str(amount))} 
                               for price, amount in data["asks"]]
                        
                        return OrderBookSnapshot(
                            bids=bids,
                            asks=asks,
                            timestamp=data["T"]  # Use exchange timestamp
                        )
                    else:
                        self.logger().warning(f"Failed to fetch order book data: {response.status}")
                        return None
        except Exception as e:
            self.logger().error(f"Error fetching order book data: {str(e)}")
            return None

    def calculate_obi(self, order_book: OrderBookSnapshot) -> tuple[float, float]:
        """Calculate Order Book Imbalance metrics"""
        bids = order_book.bids
        asks = order_book.asks
        
        bid_volume = sum(float(bid["amount"]) for bid in bids)
        ask_volume = sum(float(ask["amount"]) for ask in asks)
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0, 0
        
        obi = (bid_volume - ask_volume) / total_volume
        bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        
        return obi, bid_ask_ratio

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Get current timestamp - handle both datetime and int64 index types
        if isinstance(df.index[-1], (int, np.int64)):
            current_timestamp = int(df.index[-1])
        else:
            current_timestamp = int(df.index[-1].timestamp())
        
        # Fetch historical order book data
        order_book = await self.fetch_historical_orderbook(current_timestamp)
        
        if order_book is None:
            self.logger().warning("Failed to fetch order book data, skipping OBI calculation")
            obi, bid_ask_ratio = 0, 1  # Neutral values
        else:
            # Calculate OBI metrics
            obi, bid_ask_ratio = self.calculate_obi(order_book)
        
        # Store OBI history
        if 'obi_history' not in self.processed_data:
            self.processed_data['obi_history'] = []
        self.processed_data['obi_history'].append(obi)
        
        # Calculate OBI moving average
        obi_series = pd.Series(self.processed_data['obi_history'][-self.config.obi_window:])
        obi_ma = obi_series.mean()

        # Calculate basic indicators
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # EMAs
        ema_fast = df.ta.ema(length=self.config.ema_fast)
        ema_medium = df.ta.ema(length=self.config.ema_medium)
        ema_slow = df.ta.ema(length=self.config.ema_slow)

        # ATR and Volatility
        atr = df.ta.atr(length=self.config.atr_length)
        
        # Volume analysis
        volume_ma = volume.rolling(window=20).mean()
        volume_std = volume.rolling(window=20).std()
        volume_surge = volume > volume_ma * self.config.volume_surge
        volume_trend = volume.rolling(window=5).mean() > volume_ma

        # RSI
        rsi = df.ta.rsi(length=self.config.rsi_period)

        # Price channels
        high_band = high.rolling(10).max()
        low_band = low.rolling(10).min()

        # Trend strength
        trend_strength = abs(ema_fast - ema_slow) / ema_slow
        strong_trend = trend_strength > trend_strength.rolling(20).mean()

        # Momentum conditions
        bullish_momentum = (
            (close > close.shift(1)) & 
            (close > ema_fast) & 
            (ema_fast > ema_medium) & 
            (ema_medium > ema_slow)
        )

        bearish_momentum = (
            (close < close.shift(1)) & 
            (close < ema_fast) & 
            (ema_fast < ema_medium) & 
            (ema_medium < ema_slow)
        )

        # Modify signal generation to include OBI
        bullish_obi = (
            obi > self.config.obi_threshold and
            bid_ask_ratio > self.config.bid_ask_ratio_threshold and
            obi > obi_ma
        )

        bearish_obi = (
            obi < -self.config.obi_threshold and
            1/bid_ask_ratio > self.config.bid_ask_ratio_threshold and
            obi < obi_ma
        )

        # Combine with existing conditions
        long_condition = (
            (close > high_band.shift(1)) & 
            bullish_momentum & 
            (rsi > 50) & 
            (rsi < 70) & 
            volume_surge &  # Keep volume surge as confirmation
            volume_trend & 
            strong_trend & 
            (close > close.rolling(5).mean()) &
            bullish_obi  # Add OBI condition
        )

        short_condition = (
            (close < low_band.shift(1)) & 
            bearish_momentum & 
            (rsi < 50) & 
            (rsi > 30) & 
            volume_surge &  # Keep volume surge as confirmation
            volume_trend & 
            strong_trend & 
            (close < close.rolling(5).mean()) &
            bearish_obi  # Add OBI condition
        )

        # Apply decimation to reduce noise
        long_condition = self.decimate(long_condition)
        short_condition = self.decimate(short_condition)

        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
