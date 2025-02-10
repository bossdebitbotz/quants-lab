import asyncio
import pandas as pd
from decimal import Decimal
from typing import Dict, Optional, List
from core.data_sources import CLOBDataSource
import pandas_ta as ta
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator
from hummingbot.data_feed.candles_feed.binance_perpetual_candles import BinancePerpetualCandles

class EliteEliteConfig(DirectionalTradingControllerConfigBase):
    controller_name = "eliteelite"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = Field(
        default="1h",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    ema_short: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the short EMA period (8-13): ",
            prompt_on_new=True))
    ema_medium: int = Field(
        default=29,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the medium EMA period (21-34): ",
            prompt_on_new=True))
    ema_long: int = Field(
        default=31,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the long EMA period (34-55): ",
            prompt_on_new=True))
    macd_fast: int = Field(
        default=22,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter MACD fast period (12-26): ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=36,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter MACD slow period (26-40): ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=17,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter MACD signal period (9-17): ",
            prompt_on_new=True))
    atr_length: int = Field(
        default=3,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ATR length (3-21): ",
            prompt_on_new=True))
    atr_multiplier: float = Field(
        default=1.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ATR multiplier (1.0-2.0): ",
            prompt_on_new=True))
    orderbook_imbalance_threshold: float = Field(
        default=0.2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter orderbook imbalance threshold (0.1-0.5): ",
            prompt_on_new=True))
    volume_surge_threshold: float = Field(
        default=1.6,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter volume surge multiplier (1.2-3.0): ",
            prompt_on_new=True))
    volume_ma_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter volume MA length (10-30): ",
            prompt_on_new=True))
    use_orderbook: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda mi: "Use orderbook data for signals? (True/False): ",
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

class EliteEliteController(DirectionalTradingControllerBase):
    """
    Elite Elite Trading Strategy
    Advanced trend-following strategy with multi-timeframe analysis
    Combines EMA crossovers, MACD momentum, and orderbook imbalance
    """

    def __init__(self, config: EliteEliteConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        self.orderbook_cache = {}  # Add cache for orderbook data
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def calculate_orderbook_imbalance(self, orderbook):
        """Calculate orderbook imbalance from bid/ask pressure"""
        bids = orderbook.bids[:20]
        asks = orderbook.asks[:20]
        
        bid_pressure = sum(float(amount) for _, amount, _ in bids)
        ask_pressure = sum(float(amount) for _, amount, _ in asks)
        
        return (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)

    async def get_historical_orderbook(self, timestamp: int) -> Optional[Dict]:
        """Get historical orderbook data for a specific timestamp"""
        cache_key = f"{self.config.trading_pair}_{timestamp}"
        
        if cache_key in self.orderbook_cache:
            return self.orderbook_cache[cache_key]
            
        try:
            connector = BinancePerpetualCandles()
            orderbook_data = await connector.get_historical_orderbook(
                trading_pair=self.config.trading_pair,
                timestamp=timestamp
            )
            
            if orderbook_data is not None:
                self.orderbook_cache[cache_key] = orderbook_data
                return orderbook_data
                
        except Exception as e:
            self.logger.error(f"Error fetching historical orderbook: {e}")
        return None

    async def update_processed_data(self):
        """Process market data and generate trading signals"""
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Calculate indicators
        close = df['close']
        volume = df['volume']
        
        # Volume analysis
        volume_ma = volume.rolling(window=self.config.volume_ma_length).mean()
        volume_std = volume.rolling(window=self.config.volume_ma_length).std()
        volume_surge = volume > volume_ma * self.config.volume_surge_threshold
        volume_trend = volume.rolling(window=5).mean() > volume_ma

        # EMAs
        ema_short = df.ta.ema(length=self.config.ema_short)
        ema_medium = df.ta.ema(length=self.config.ema_medium)
        ema_long = df.ta.ema(length=self.config.ema_long)

        # MACD
        macd = df.ta.macd(
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )
        macd_hist = macd[f'MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']

        # ATR and support/resistance
        atr = df.ta.atr(length=self.config.atr_length)
        atr_support = close.shift(1) - atr * self.config.atr_multiplier
        atr_resistance = close.shift(1) + atr * self.config.atr_multiplier

        # Get historical orderbook data if enabled
        if self.config.use_orderbook:
            current_timestamp = int(df.index[-1].timestamp())
            try:
                orderbook_data = await self.get_historical_orderbook(current_timestamp)
                if orderbook_data is not None:
                    imbalance = self.calculate_orderbook_imbalance(orderbook_data)
                else:
                    raise ValueError("No orderbook data available")
            except Exception as e:
                self.logger.error(f"Error processing orderbook data: {e}")
                # Fallback to price momentum
                price_change = (close - close.shift(1)) / close.shift(1)
                price_ma = price_change.rolling(window=5).mean()
                imbalance = price_ma.iloc[-1]
        else:
            # Use price momentum directly if orderbook disabled
            price_change = (close - close.shift(1)) / close.shift(1)
            price_ma = price_change.rolling(window=5).mean()
            imbalance = price_ma.iloc[-1]

        # Generate signals
        df['signal'] = 0

        long_condition = (
            (ema_short > ema_medium) &
            (ema_medium > ema_long) &
            (close > ema_short) &
            (macd_hist > 0) &
            (abs(imbalance) > self.config.orderbook_imbalance_threshold) &
            volume_surge &
            volume_trend
        )

        short_condition = (
            (ema_short < ema_medium) &
            (ema_medium < ema_long) &
            (close < ema_short) &
            (macd_hist < 0) &
            (abs(imbalance) > self.config.orderbook_imbalance_threshold) &
            volume_surge &
            volume_trend
        )

        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1

        # Store processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        self.processed_data["metrics"] = {
            "imbalance": imbalance,
            "macd_hist": macd_hist.iloc[-1],
            "atr_support": atr_support.iloc[-1],
            "atr_resistance": atr_resistance.iloc[-1],
            "volume_surge": volume_surge.iloc[-1],
            "volume_trend": volume_trend.iloc[-1],
            "volume_ma": volume_ma.iloc[-1],
            "current_volume": volume.iloc[-1]
        }

