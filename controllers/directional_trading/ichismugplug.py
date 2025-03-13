from typing import List, Tuple

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator
import pandas as pd


def calculate_zlema(series, length):
    """Calculate Zero-Lag Exponential Moving Average (ZLEMA)
    
    Args:
        series: Price series to calculate ZLEMA for
        length: Lookback period
        
    Returns:
        Pandas Series containing ZLEMA values
    """
    lag = (length - 1) // 2
    zero_lag = series + (series.diff(lag))
    zlema = zero_lag.ewm(span=length, adjust=False).mean()
    return zlema


def calculate_kst(df: pd.DataFrame, 
                 roc1_period: int, roc2_period: int, 
                 roc3_period: int, roc4_period: int,
                 ma1_period: int, ma2_period: int, 
                 ma3_period: int, ma4_period: int,
                 signal_period: int) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Know Sure Thing (KST) indicator
    
    Args:
        df: DataFrame with price data
        roc1_period to roc4_period: Periods for Rate of Change calculations
        ma1_period to ma4_period: Periods for moving averages of ROCs
        signal_period: Period for signal line
    
    Returns:
        Tuple of (KST line, Signal line)
    """
    # Calculate ROC values
    roc1 = df['close'].pct_change(roc1_period)
    roc2 = df['close'].pct_change(roc2_period) 
    roc3 = df['close'].pct_change(roc3_period)
    roc4 = df['close'].pct_change(roc4_period)
    
    # Calculate smoothed ROC values
    roc1_ma = roc1.rolling(window=ma1_period).mean()
    roc2_ma = roc2.rolling(window=ma2_period).mean()
    roc3_ma = roc3.rolling(window=ma3_period).mean()
    roc4_ma = roc4.rolling(window=ma4_period).mean()
    
    # Calculate KST
    kst = (roc1_ma * 1) + (roc2_ma * 2) + (roc3_ma * 3) + (roc4_ma * 4)
    
    # Calculate signal line
    signal = kst.rolling(window=signal_period).mean()
    
    return kst, signal


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
    zlema_short: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ZLEMA short period: ",
            prompt_on_new=True))
    zlema_medium: int = Field(
        default=29,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ZLEMA medium period: ",
            prompt_on_new=True))
    zlema_long: int = Field(
        default=31,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ZLEMA long period: ",
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
    kst_roc1_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST ROC1 period: ",
            prompt_on_new=True))
    kst_roc2_period: int = Field(
        default=15,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST ROC2 period: ",
            prompt_on_new=True))
    kst_roc3_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST ROC3 period: ",
            prompt_on_new=True))
    kst_roc4_period: int = Field(
        default=30,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST ROC4 period: ",
            prompt_on_new=True))
    kst_ma1_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST MA1 period: ",
            prompt_on_new=True))
    kst_ma2_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST MA2 period: ",
            prompt_on_new=True))
    kst_ma3_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST MA3 period: ",
            prompt_on_new=True))
    kst_ma4_period: int = Field(
        default=15,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST MA4 period: ",
            prompt_on_new=True))
    kst_signal_period: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the KST signal period: ",
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
        
        # Initialize executors list
        self.executors = []

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
        
        # Calculate ZLEMA using our helper function
        df[f"ZLEMA_{self.config.zlema_short}"] = calculate_zlema(df["close"], self.config.zlema_short)
        df[f"ZLEMA_{self.config.zlema_medium}"] = calculate_zlema(df["close"], self.config.zlema_medium)
        df[f"ZLEMA_{self.config.zlema_long}"] = calculate_zlema(df["close"], self.config.zlema_long)

        # Add Ichimoku Cloud
        ichimoku = df.ta.ichimoku(
            tenkan=self.config.tenkan_period,
            kijun=self.config.kijun_period, 
            senkou=self.config.senkou_span_b_period,
            displacement=self.config.displacement,
            append=True
        )

        # Get the actual column names based on actual pandas-ta output
        tenkan_col = f"ISA_{self.config.tenkan_period}"    # Conversion line (Tenkan-sen)
        kijun_col = f"ISB_{self.config.kijun_period}"      # Base line (Kijun-sen)
        senkou_a_col = f"ITS_{self.config.tenkan_period}"  # Leading Span A (Senkou Span A)
        senkou_b_col = f"IKS_{self.config.kijun_period}"   # Leading Span B (Senkou Span B)
        chikou_col = f"ICS_{self.config.kijun_period}"     # Lagging Span (Chikou Span)

        # For debugging, you might want to add this:
        if not all(col in df.columns for col in [tenkan_col, kijun_col, senkou_a_col, senkou_b_col, chikou_col]):
            available_cols = df.columns.tolist()
            raise ValueError(f"Missing Ichimoku columns. Looking for: {[tenkan_col, kijun_col, senkou_a_col, senkou_b_col, chikou_col]}, Available columns: {available_cols}")

        # Get Ichimoku values using correct column names
        tenkan = df[tenkan_col]     # Conversion line
        kijun = df[kijun_col]       # Base line
        senkou_a = df[senkou_a_col] # Leading Span A
        senkou_b = df[senkou_b_col] # Leading Span B
        chikou = df[chikou_col]     # Lagging Span

        # Add volume indicators
        df[f"volume_ma_{self.config.volume_ma_period}"] = df["volume"].rolling(
            window=self.config.volume_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df[f"volume_ma_{self.config.volume_ma_period}"]

        # Calculate ATR bands
        df["long_atr_support"] = df["close"].shift(1) - df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        df["short_atr_resistance"] = df["close"].shift(1) + df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier

        # Get indicator values
        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        short_zlema = df[f"ZLEMA_{self.config.zlema_short}"]
        medium_zlema = df[f"ZLEMA_{self.config.zlema_medium}"]
        long_zlema = df[f"ZLEMA_{self.config.zlema_long}"]
        close = df["close"]
        volume = df["volume"]
        volume_ma = df[f"volume_ma_{self.config.volume_ma_period}"]
        volume_ratio = volume / volume_ma

        # Add KST calculation
        kst_line, kst_signal = calculate_kst(
            df,
            self.config.kst_roc1_period,
            self.config.kst_roc2_period,
            self.config.kst_roc3_period,
            self.config.kst_roc4_period,
            self.config.kst_ma1_period,
            self.config.kst_ma2_period,
            self.config.kst_ma3_period,
            self.config.kst_ma4_period,
            self.config.kst_signal_period
        )
        
        df['kst'] = kst_line
        df['kst_signal'] = kst_signal
        df['kst_hist'] = df['kst'] - df['kst_signal']

        # Calculate Fibonacci levels manually with shorter window for more frequent signals
        window = 10  # Reduced from 20 to be more responsive
        high_max = df['high'].rolling(window=window).max()
        low_min = df['low'].rolling(window=window).min()
        price_range = high_max - low_min
        
        # Calculate Fibonacci retracement levels
        df['FIBO_0'] = low_min  # 0%
        df['FIBO_236'] = low_min + (price_range * 0.236)  # 23.6%
        df['FIBO_382'] = low_min + (price_range * 0.382)  # 38.2%
        df['FIBO_500'] = low_min + (price_range * 0.500)  # 50.0%
        df['FIBO_618'] = low_min + (price_range * 0.618)  # 61.8%
        df['FIBO_786'] = low_min + (price_range * 0.786)  # 78.6%
        df['FIBO_1'] = high_max  # 100%

        # Add dynamic buffer for Fibonacci levels
        fib_buffer = price_range * 0.01  # 1% buffer
        
        # Enhanced signal generation with more flexible Fibonacci conditions
        long_condition = (
            (short_zlema > medium_zlema) & 
            (medium_zlema > long_zlema) & 
            (close > short_zlema) & 
            (close > df["long_atr_support"]) & 
            (macdh > 0) &
            (close > tenkan) &
            (close > kijun) &
            (senkou_a > senkou_b) &
            (volume > volume_ma) &  
            (volume_ratio > 1.5) &
            (df['kst'] > df['kst_signal']) &
            (df['kst_hist'] > 0) &
            # Relaxed Fibonacci conditions for long
            (
                # Either price is near or above 23.6% level
                ((close >= df['FIBO_236'] - fib_buffer) & (close <= df['FIBO_382'] + fib_buffer)) |
                # Or near or above 38.2% level
                ((close >= df['FIBO_382'] - fib_buffer) & (close <= df['FIBO_500'] + fib_buffer)) |
                # Or near or above 50% level
                ((close >= df['FIBO_500'] - fib_buffer) & (close <= df['FIBO_618'] + fib_buffer))
            )
        )

        short_condition = (
            (short_zlema < medium_zlema) & 
            (medium_zlema < long_zlema) & 
            (close < short_zlema) & 
            (close < df["short_atr_resistance"]) & 
            (macdh < 0) &
            (close < tenkan) &
            (close < kijun) &
            (senkou_a < senkou_b) &
            (volume > volume_ma) &  
            (volume_ratio > 1.5) &
            (df['kst'] < df['kst_signal']) &
            (df['kst_hist'] < 0) &
            # Relaxed Fibonacci conditions for short
            (
                # Either price is near or below 78.6% level
                ((close <= df['FIBO_786'] + fib_buffer) & (close >= df['FIBO_618'] - fib_buffer)) |
                # Or near or below 61.8% level
                ((close <= df['FIBO_618'] + fib_buffer) & (close >= df['FIBO_500'] - fib_buffer)) |
                # Or near or below 50% level
                ((close <= df['FIBO_500'] + fib_buffer) & (close >= df['FIBO_382'] - fib_buffer))
            )
        )

        # Set signals
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        
        # Ensure executor configuration is properly set
        if self.processed_data["signal"] != 0 and not self.executors:
            side = "BUY" if self.processed_data["signal"] > 0 else "SELL"
            self.executors.append({
                "config": {
                    "side": side,
                    "trading_pair": self.config.trading_pair,
                    "connector_name": self.config.connector_name,
                    "amount": self.config.total_amount_quote,
                    "take_profit": self.config.take_profit,
                    "stop_loss": self.config.stop_loss,
                    "trailing_stop": self.config.trailing_stop,
                }
            })

    async def get_executors_status(self):
        return self.executors
