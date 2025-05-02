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


class ESPFibPivConfig(DirectionalTradingControllerConfigBase):
    controller_name = "esp_fib_piv"
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
    fib_pivot_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Fibonacci pivot calculation period (10-20): ",
            prompt_on_new=True))
    fib_confirmation_threshold: float = Field(
        default=0.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Fibonacci confirmation threshold (0.3-0.7): ",
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


class ESPFibPivController(DirectionalTradingControllerBase):
    """
    SmugPlug Emerald Elite v3 Lite
    High-Precision Breakout Detection with Multi-Timeframe Confirmation
    A lightweight version optimized for performance
    Enhanced with Fibonacci pivot points for improved support/resistance detection
    """

    def __init__(self, config: ESPFibPivConfig, *args, **kwargs):
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
            
    def calculate_fibonacci_pivots(self, df, period=14):
        """
        Calculate Fibonacci pivot points based on previous period's high, low, and close.
        
        Returns a DataFrame with pivot point (PP) and support/resistance levels:
        - S3, S2, S1: Support levels
        - R1, R2, R3: Resistance levels
        """
        pivots = pd.DataFrame(index=df.index)
        
        # Fibonacci ratios
        fib_ratios = {
            'R3': 1.618,  # 161.8%
            'R2': 1.272,  # 127.2%
            'R1': 0.618,  # 61.8%
            'S1': 0.382,  # 38.2%
            'S2': 0.272,  # 27.2%
            'S3': 0.118   # 11.8%
        }
        
        # Calculate pivot points for each row based on previous periods
        for i in range(period, len(df)):
            # Get high, low, close for the period
            high_period = df['high'].iloc[i-period:i].max()
            low_period = df['low'].iloc[i-period:i].min()
            close_period = df['close'].iloc[i-1]
            
            # Calculate pivot point (PP)
            pp = (high_period + low_period + close_period) / 3
            pivots.loc[df.index[i], 'PP'] = pp
            
            # Calculate Fibonacci support and resistance levels
            range_hl = high_period - low_period
            
            # Resistance levels
            pivots.loc[df.index[i], 'R1'] = pp + (range_hl * fib_ratios['R1'])
            pivots.loc[df.index[i], 'R2'] = pp + (range_hl * fib_ratios['R2'])
            pivots.loc[df.index[i], 'R3'] = pp + (range_hl * fib_ratios['R3'])
            
            # Support levels
            pivots.loc[df.index[i], 'S1'] = pp - (range_hl * fib_ratios['S1'])
            pivots.loc[df.index[i], 'S2'] = pp - (range_hl * fib_ratios['S2'])
            pivots.loc[df.index[i], 'S3'] = pp - (range_hl * fib_ratios['S3'])
        
        return pivots

    def calculate_fibonacci_retracements(self, df, lookback=20):
        """
        Calculate Fibonacci retracement levels for trend continuation trades.
        """
        retracements = pd.DataFrame(index=df.index)
        
        # Fibonacci retracement levels
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for i in range(lookback, len(df)):
            # Find swing high and low in the lookback period
            window = df.iloc[i-lookback:i]
            swing_high = window['high'].max()
            swing_high_idx = window['high'].idxmax()
            swing_low = window['low'].min()
            swing_low_idx = window['low'].idxmin()
            
            # Determine trend direction based on which came first
            if df.index.get_loc(swing_low_idx) < df.index.get_loc(swing_high_idx):
                # Uptrend: low to high
                price_range = swing_high - swing_low
                for level in fib_levels:
                    level_name = f'Ret_{int(level*1000)}'
                    retracements.loc[df.index[i], level_name] = swing_high - (level * price_range)
                retracements.loc[df.index[i], 'trend'] = 1  # Mark as uptrend
            else:
                # Downtrend: high to low
                price_range = swing_high - swing_low
                for level in fib_levels:
                    level_name = f'Ret_{int(level*1000)}'
                    retracements.loc[df.index[i], level_name] = swing_low + (level * price_range)
                retracements.loc[df.index[i], 'trend'] = -1  # Mark as downtrend
        
        return retracements

    def calculate_fibonacci_extensions(self, df, lookback=20):
        """
        Calculate Fibonacci extension levels for breakout targets.
        """
        extensions = pd.DataFrame(index=df.index)
        
        # Fibonacci extension levels
        ext_levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
        
        for i in range(lookback, len(df)):
            # Find swing high and low in the lookback period
            window = df.iloc[i-lookback:i]
            swing_high = window['high'].max()
            swing_high_idx = window['high'].idxmax()
            swing_low = window['low'].min()
            swing_low_idx = window['low'].idxmin()
            
            # Determine trend direction based on which came first
            if df.index.get_loc(swing_low_idx) < df.index.get_loc(swing_high_idx):
                # Uptrend: low to high
                price_range = swing_high - swing_low
                for level in ext_levels:
                    level_name = f'Ext_{int(level*1000)}'
                    extensions.loc[df.index[i], level_name] = swing_high + (level * price_range)
                extensions.loc[df.index[i], 'trend'] = 1  # Mark as uptrend
            else:
                # Downtrend: high to low
                price_range = swing_high - swing_low
                for level in ext_levels:
                    level_name = f'Ext_{int(level*1000)}'
                    extensions.loc[df.index[i], level_name] = swing_low - (level * price_range)
                extensions.loc[df.index[i], 'trend'] = -1  # Mark as downtrend
        
        return extensions

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

        # Determine overall market trend
        market_trend = pd.Series(0, index=df.index)
        market_trend[(ema_fast > ema_medium) & (ema_medium > ema_slow)] = 1  # Uptrend
        market_trend[(ema_fast < ema_medium) & (ema_medium < ema_slow)] = -1  # Downtrend
        
        # Calculate all Fibonacci levels
        fib_pivots = self.calculate_fibonacci_pivots(df, period=self.config.fib_pivot_period)
        fib_retracements = self.calculate_fibonacci_retracements(df, lookback=self.config.fib_pivot_period)
        fib_extensions = self.calculate_fibonacci_extensions(df, lookback=self.config.fib_pivot_period)
        
        # Add Fibonacci levels to dataframe
        for col in fib_pivots.columns:
            df[f'fib_{col}'] = fib_pivots[col]
        
        for col in fib_retracements.columns:
            df[f'{col}'] = fib_retracements[col]
            
        for col in fib_extensions.columns:
            df[f'{col}'] = fib_extensions[col]

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
        
        # ===== BREAKOUT SIGNALS =====
        
        # Breakout above resistance with volume confirmation
        breakout_long = (
            (close > df['fib_R1']) &  # Price breaks above R1 resistance
            (close.shift(1) <= df['fib_R1']) &  # Previous close was below R1
            (volume > volume_ma * 1.5) &  # Volume surge confirms breakout
            (rsi > 50) &  # RSI confirms bullish momentum
            (market_trend == 1)  # Overall trend is up
        )
        
        # Breakout below support with volume confirmation
        breakout_short = (
            (close < df['fib_S1']) &  # Price breaks below S1 support
            (close.shift(1) >= df['fib_S1']) &  # Previous close was above S1
            (volume > volume_ma * 1.5) &  # Volume surge confirms breakout
            (rsi < 50) &  # RSI confirms bearish momentum
            (market_trend == -1)  # Overall trend is down
        )
        
        # ===== TREND CONTINUATION SIGNALS =====
        
        # Trend continuation - buying dips in uptrend
        continuation_long = pd.Series(False, index=df.index)
        # Trend continuation - selling rallies in downtrend
        continuation_short = pd.Series(False, index=df.index)
        
        for i in range(len(df)):
            if i < self.config.fib_pivot_period:
                continue
            
            current_price = close.iloc[i]
            prev_price = close.iloc[i-1]
            
            # For long signals in uptrend: price bounced off key retracement level
            if fib_retracements.loc[df.index[i], 'trend'] == 1:  # In uptrend
                # Check if price previously dipped to retracement level and is now bouncing up
                ret_382 = fib_retracements.loc[df.index[i], 'Ret_382']
                ret_500 = fib_retracements.loc[df.index[i], 'Ret_500']
                ret_618 = fib_retracements.loc[df.index[i], 'Ret_618']
                
                # Price bounced off 38.2%, 50% or 61.8% retracement
                if ((prev_price <= ret_382 * 1.005 and current_price > ret_382) or
                    (prev_price <= ret_500 * 1.005 and current_price > ret_500) or
                    (prev_price <= ret_618 * 1.005 and current_price > ret_618)):
                    # Additional confirmation: RSI was oversold and is now rising
                    if rsi.iloc[i-1] < 40 and rsi.iloc[i] > rsi.iloc[i-1]:
                        continuation_long.iloc[i] = True
            
            # For short signals in downtrend: price bounced off key retracement level
            elif fib_retracements.loc[df.index[i], 'trend'] == -1:  # In downtrend
                # Check if price previously rose to retracement level and is now dropping
                ret_382 = fib_retracements.loc[df.index[i], 'Ret_382']
                ret_500 = fib_retracements.loc[df.index[i], 'Ret_500']
                ret_618 = fib_retracements.loc[df.index[i], 'Ret_618']
                
                # Price bounced off 38.2%, 50% or 61.8% retracement
                if ((prev_price >= ret_382 * 0.995 and current_price < ret_382) or
                    (prev_price >= ret_500 * 0.995 and current_price < ret_500) or
                    (prev_price >= ret_618 * 0.995 and current_price < ret_618)):
                    # Additional confirmation: RSI was overbought and is now falling
                    if rsi.iloc[i-1] > 60 and rsi.iloc[i] < rsi.iloc[i-1]:
                        continuation_short.iloc[i] = True
        
        # Combine breakout and continuation signals
        long_condition = (breakout_long | continuation_long) & bullish_momentum & strong_trend
        short_condition = (breakout_short | continuation_short) & bearish_momentum & strong_trend
        
        # Apply decimation to reduce noise
        long_condition = self.decimate(long_condition)
        short_condition = self.decimate(short_condition)

        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Add signal type for analysis
        df['signal_type'] = 'none'
        df.loc[breakout_long & long_condition, 'signal_type'] = 'breakout_long'
        df.loc[breakout_short & short_condition, 'signal_type'] = 'breakout_short'
        df.loc[continuation_long & long_condition, 'signal_type'] = 'continuation_long'
        df.loc[continuation_short & short_condition, 'signal_type'] = 'continuation_short'

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        
        #