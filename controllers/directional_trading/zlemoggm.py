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
import numpy as np


class ZLEMAConfig(DirectionalTradingControllerConfigBase):
    controller_name = "zlemoggm"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="1m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    source: str = Field(
        default="hlc3",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the price source (open, high, low, close, hlc3, ohlc4): ",
            prompt_on_new=True))
    period_fast: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the fast ZLEMA period (1-20): ",
            prompt_on_new=True))
    period_medium: int = Field(
        default=21,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the medium ZLEMA period (15-30): ",
            prompt_on_new=True))
    period_slow: int = Field(
        default=55,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the slow ZLEMA period (30-100): ",
            prompt_on_new=True))
    kalman_value1_coef: float = Field(
        default=0.2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Kalman filter value1 coefficient (0.0-1.0): ",
            prompt_on_new=True))
    kalman_value2_coef: float = Field(
        default=0.1,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Kalman filter value2 coefficient (0.0-1.0): ",
            prompt_on_new=True))
    kalman_default_alpha: float = Field(
        default=0.1,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Kalman filter default alpha (0.0-1.0): ",
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


class ZLEMAController(DirectionalTradingControllerBase):
    """
    Zero Lag EMA with Kalman filter Controller
    Based on the ZLEMA indicator by M0rty
    """

    def __init__(self, config: ZLEMAConfig, *args, **kwargs):
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

    def get_price_source(self, df: pd.DataFrame) -> pd.Series:
        """Get the price source based on config"""
        if self.config.source == "open":
            return df["open"]
        elif self.config.source == "high":
            return df["high"]
        elif self.config.source == "low":
            return df["low"]
        elif self.config.source == "close":
            return df["close"]
        elif self.config.source == "hlc3":
            return (df["high"] + df["low"] + df["close"]) / 3
        elif self.config.source == "ohlc4":
            return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        else:
            return df["close"]

    def kalman_filter(self, src: pd.Series) -> pd.Series:
        """
        Implement Kalman filter based on Ehlers Optimal Tracking Filters
        https://www.dimensionetrading.com/Pattern_e_indicatori/Ehlers%20_Optimal%20Tracking%20Filters_.pdf
        """
        # Calculate True Range for volatility estimation
        high = src.index.map(lambda x: src.loc[:x].iloc[-1] if x in src.index else None)
        low = src.index.map(lambda x: src.loc[:x].iloc[-1] if x in src.index else None)
        tr = pd.Series(high - low)
        
        # Initialize result series
        result = pd.Series(index=src.index, dtype=float)
        
        # Initial values
        value1 = 0.0
        value2 = 0.0
        value3 = 0.0
        
        # Process each data point
        for i, (idx, price) in enumerate(src.items()):
            if i > 0:
                prev_price = src.iloc[i-1]
                # Use config values for coefficients
                value1 = self.config.kalman_value1_coef * (price - prev_price) + (1 - self.config.kalman_value1_coef) * value1
                value2 = self.config.kalman_value2_coef * tr.iloc[i] + (1 - self.config.kalman_value2_coef) * value2
                
                # Calculate lambda and alpha
                if value2 != 0:
                    lambda_val = abs(value1 / value2)
                    alpha = (-pow(lambda_val, 2) + np.sqrt(pow(lambda_val, 4) + 16 * pow(lambda_val, 2))) / 8
                else:
                    # Use config value for default alpha
                    alpha = self.config.kalman_default_alpha
                
                # Apply filter
                value3 = alpha * price + (1 - alpha) * value3
            else:
                # Initialize with first value
                value3 = price
            
            result.iloc[i] = value3
        
        return result

    def zlema(self, src: pd.Series, period: int) -> pd.Series:
        """
        Zero Lag Exponential Moving Average implementation
        https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
        """
        # Calculate lag
        lag = int((period - 1) / 2)
        
        # Create ema_data
        ema_data = src + (src - src.shift(lag))
        
        # Calculate EMA of ema_data
        return ema_data.ewm(span=period, adjust=False).mean()

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Get price source
        src = self.get_price_source(df)
        
        # Apply Kalman filter (now mandatory)
        src = self.kalman_filter(src)
        
        # Calculate ZLEMAs
        ma1 = self.zlema(src, self.config.period_fast)
        ma2 = self.zlema(src, self.config.period_medium)
        ma3 = self.zlema(src, self.config.period_slow)
        
        # Store calculated values
        df['zlema_fast'] = ma1
        df['zlema_medium'] = ma2
        df['zlema_slow'] = ma3
        
        # Generate signals based on crosses (mandatory)
        # Crossover (buy signal)
        df['buy_signal'] = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
        
        # Crossunder (sell signal)
        df['sell_signal'] = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))
        
        # Set signal values
        df['signal'] = 0
        df.loc[df['buy_signal'], 'signal'] = 1
        df.loc[df['sell_signal'], 'signal'] = -1
        
        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1] if not df["signal"].empty else 0
        self.processed_data["features"] = df