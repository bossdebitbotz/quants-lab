from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator


class IchimokuSmugPlugControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "ichismugplug"
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
    macd_fast: int = Field(
        default=22,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=36,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=17,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    ema_short: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA short period: ",
            prompt_on_new=True))
    ema_medium: int = Field(
        default=29,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA medium period: ",
            prompt_on_new=True))
    ema_long: int = Field(
        default=31,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA long period: ",
            prompt_on_new=True))
    atr_length: int = Field(
        default=11,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR length: ",
            prompt_on_new=True))
    atr_multiplier: float = Field(
        default=1.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR multiplier: ",
            prompt_on_new=True))
    tenkan_period: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Tenkan-sen period: ",
            prompt_on_new=True))
    kijun_period: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Kijun-sen period: ",
            prompt_on_new=True))
    senkou_span_b_period: int = Field(
        default=52,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Senkou Span B period: ",
            prompt_on_new=True))
    displacement: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the displacement period: ",
            prompt_on_new=True))
    volume_ma_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume MA period: ",
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


class IchimokuSmugPlugController(DirectionalTradingControllerBase):

    def __init__(self, config: IchimokuSmugPlugControllerConfig, *args, **kwargs):
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

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        # Add indicators and retrieve actual column names
        macd_columns = df.ta.macd(
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            append=True
        )
        atr_columns = df.ta.atr(length=self.config.atr_length, append=True)
        ema_short_columns = df.ta.ema(length=self.config.ema_short, append=True)
        ema_medium_columns = df.ta.ema(length=self.config.ema_medium, append=True)
        ema_long_columns = df.ta.ema(length=self.config.ema_long, append=True)

        # Add Ichimoku Cloud
        ichimoku_columns = df.ta.ichimoku(
            tenkan=self.config.tenkan_period,
            kijun=self.config.kijun_period,
            senkou=self.config.senkou_span_b_period,
            displacement=self.config.displacement,
            append=True
        )

        # Add volume indicators
        df[f"volume_ma_{self.config.volume_ma_period}"] = df["volume"].rolling(
            window=self.config.volume_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df[f"volume_ma_{self.config.volume_ma_period}"]

        # Calculate ATR bands
        df["long_atr_support"] = df["close"].shift(1) - df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        df["short_atr_resistance"] = df["close"].shift(1) + df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier

        # Get indicator values
        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        short_ema = df[f"EMA_{self.config.ema_short}"]
        medium_ema = df[f"EMA_{self.config.ema_medium}"]
        long_ema = df[f"EMA_{self.config.ema_long}"]
        close = df["close"]
        volume = df["volume"]
        volume_ma = df[f"volume_ma_{self.config.volume_ma_period}"]
        volume_ratio = volume / volume_ma

        # Get Ichimoku values
        isa = df["ISA_9"]  # Conversion line (Tenkan-sen)
        isb = df["ISB_26"]  # Base line (Kijun-sen)
        its = df["ITS_9"]  # Leading Span A (Senkou Span A)
        iks = df["IKS_26"]  # Leading Span B (Senkou Span B)

        # Generate signals based on MACD, EMAs, ATR, Ichimoku and Volume
        long_condition = (
            (short_ema > medium_ema) & 
            (medium_ema > long_ema) & 
            (close > short_ema) & 
            (close > df["long_atr_support"]) & 
            (macdh > 0) &
            (close > isa) &  # Price above conversion line
            (close > isb) &  # Price above base line
            (its > iks) &    # Leading Span A above Leading Span B (bullish cloud)
            (volume > volume_ma) &  # Volume above average
            (volume_ratio > 1.5)    # Volume spike for breakout confirmation
        )

        short_condition = (
            (short_ema < medium_ema) & 
            (medium_ema < long_ema) & 
            (close < short_ema) & 
            (close < df["short_atr_resistance"]) & 
            (macdh < 0) &
            (close < isa) &  # Price below conversion line
            (close < isb) &  # Price below base line
            (its < iks) &    # Leading Span A below Leading Span B (bearish cloud)
            (volume > volume_ma) &  # Volume above average
            (volume_ratio > 1.5)    # Volume spike for breakout confirmation
        )

        # Set signals
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
