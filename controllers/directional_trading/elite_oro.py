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
import asyncio
import contextlib


class EliteOroConfig(DirectionalTradingControllerConfigBase):
    controller_name = "elite_oro"
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
    zlema_fast: int = Field(
        default=4,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the fast ZLEMA period (3-8): ",
            prompt_on_new=True))
    zlema_medium: int = Field(
        default=12,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the medium ZLEMA period (8-21): ",
            prompt_on_new=True))
    zlema_slow: int = Field(
        default=13,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the slow ZLEMA period (13-34): ",
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
    volume_ma_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume MA period: ",
            prompt_on_new=True))
    volume_surge: float = Field(
        default=1.6,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume surge multiplier (1.5-3.0): ",
            prompt_on_new=True))
    price_channel_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the price channel period (5-20): ",
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


class EliteOroController(DirectionalTradingControllerBase):
    """
    Elite Oro Strategy
    High-Precision Breakout Detection with ZLEMA Confirmation
    Combines the best elements of Oro and SmugPlug strategies
    Optimized for production execution with enhanced signal quality
    """

    def __init__(self, config: EliteOroConfig, *args, **kwargs):
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

    def calculate_zlema(self, data, length):
        """
        Calculate Zero Lag EMA manually since pandas_ta doesn't have it built-in.
        ZLEMA = EMA(2*Close - Close(lag)) where lag = (length-1)/2
        """
        lag = int((length - 1) / 2)
        data_lag = data.shift(lag)
        data_zlema = 2 * data - data_lag
        return data_zlema.ewm(span=length, adjust=False).mean()

    def decimate(self, series):
        """Reduce signal noise by identifying unique signal changes."""
        start_of_sequence = series & ~series.shift(fill_value=False)
        group_number = start_of_sequence.cumsum()
        return series & (group_number == group_number.where(series).groupby(
            group_number).transform('first'))

    async def update_processed_data(self):
        try:
            df = self.market_data_provider.get_candles_df(
                connector_name=self.config.candles_connector,
                trading_pair=self.config.candles_trading_pair,
                interval=self.config.interval,
                max_records=self.max_records
            )
            
            if df is None or df.empty:
                return
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, unit='s')
            
            # Calculate basic indicators
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            try:
                # Zero Lag EMAs using custom implementation
                df['ZLEMA_fast'] = self.calculate_zlema(close, self.config.zlema_fast)
                df['ZLEMA_medium'] = self.calculate_zlema(close, self.config.zlema_medium)
                df['ZLEMA_slow'] = self.calculate_zlema(close, self.config.zlema_slow)
                
                # ATR for volatility measurement
                df.ta.atr(length=self.config.atr_length, append=True)
                
                # Volume analysis
                df[f"volume_ma_{self.config.volume_ma_period}"] = volume.rolling(
                    window=self.config.volume_ma_period).mean()
                df["volume_std"] = volume.rolling(window=self.config.volume_ma_period).std()
                df["volume_z_score"] = (volume - df[f"volume_ma_{self.config.volume_ma_period}"]) / df["volume_std"]
                df["volume_surge"] = volume > df[f"volume_ma_{self.config.volume_ma_period}"] * self.config.volume_surge
                df["volume_trend"] = volume.rolling(window=5).mean() > df[f"volume_ma_{self.config.volume_ma_period}"]
                
                # Price channels
                df["high_band"] = high.rolling(self.config.price_channel_period).max()
                df["low_band"] = low.rolling(self.config.price_channel_period).min()
                
                # RSI for momentum confirmation (from SmugPlug)
                df["rsi"] = df.ta.rsi(length=self.config.rsi_period)
                
                # Trend strength
                df["trend_strength"] = abs(df['ZLEMA_fast'] - df['ZLEMA_slow']) / df['ZLEMA_slow']
                df["strong_trend"] = df["trend_strength"] > df["trend_strength"].rolling(20).mean()
                
                # Volatility measurement
                df["atr_volatility"] = df[f"ATRr_{self.config.atr_length}"] / df["close"] * 100
                
                # Momentum conditions (from SmugPlug)
                df["bullish_momentum"] = (
                    (close > close.shift(1)) & 
                    (close > df['ZLEMA_fast']) & 
                    (df['ZLEMA_fast'] > df['ZLEMA_medium']) & 
                    (df['ZLEMA_medium'] > df['ZLEMA_slow'])
                )
                
                df["bearish_momentum"] = (
                    (close < close.shift(1)) & 
                    (close < df['ZLEMA_fast']) & 
                    (df['ZLEMA_fast'] < df['ZLEMA_medium']) & 
                    (df['ZLEMA_medium'] < df['ZLEMA_slow'])
                )
                
                # Signal generation
                df['signal'] = 0
                
                # Combined conditions from both strategies
                long_condition = (
                    (close > df["high_band"].shift(1)) &  # Breakout above previous high
                    df["bullish_momentum"] &  # Momentum confirmation
                    (df["rsi"] > 50) &  # RSI confirmation
                    (df["rsi"] < 70) &  # Not overbought
                    df["volume_surge"] &  # Volume surge
                    df["volume_trend"] &  # Volume trend
                    df["strong_trend"] &  # Strong trend
                    (close > close.rolling(5).mean())  # Price above short-term MA
                )
                
                short_condition = (
                    (close < df["low_band"].shift(1)) &  # Breakout below previous low
                    df["bearish_momentum"] &  # Momentum confirmation
                    (df["rsi"] < 50) &  # RSI confirmation
                    (df["rsi"] > 30) &  # Not oversold
                    df["volume_surge"] &  # Volume surge
                    df["volume_trend"] &  # Volume trend
                    df["strong_trend"] &  # Strong trend
                    (close < close.rolling(5).mean())  # Price below short-term MA
                )
                
                # Apply decimation to reduce noise
                long_condition = self.decimate(long_condition)
                short_condition = self.decimate(short_condition)
                
                df.loc[long_condition, 'signal'] = 1
                df.loc[short_condition, 'signal'] = -1
                
                # Update processed data
                self.processed_data["signal"] = df["signal"].iloc[-1]
                self.processed_data["features"] = df
                
            except Exception as e:
                raise
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise
