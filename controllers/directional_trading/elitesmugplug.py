from typing import List
import pandas as pd
import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator


class SmugPlug3LiteConfig(DirectionalTradingControllerConfigBase):
    controller_name = "elitesmugplug"
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


class SmugPlug3LiteController(DirectionalTradingControllerBase):
    """
    SmugPlug Emerald Elite v3 Lite
    High-Precision Breakout Detection with Multi-Timeframe Confirmation
    A lightweight version optimized for performance
    """

    def __init__(self, config: SmugPlug3LiteConfig, *args, **kwargs):
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

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

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

        # Signal generation
        df['signal'] = 0

        long_condition = (
            (close > high_band.shift(1)) & 
            bullish_momentum & 
            (rsi > 50) & 
            (rsi < 70) & 
            volume_surge & 
            volume_trend & 
            strong_trend & 
            (close > close.rolling(5).mean())
        )

        short_condition = (
            (close < low_band.shift(1)) & 
            bearish_momentum & 
            (rsi < 50) & 
            (rsi > 30) & 
            volume_surge & 
            volume_trend & 
            strong_trend & 
            (close < close.rolling(5).mean())
        )

        # Apply decimation to reduce noise
        long_condition = self.decimate(long_condition)
        short_condition = self.decimate(short_condition)

        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
