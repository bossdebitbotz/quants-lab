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
    bb_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands period (15-30): ",
            prompt_on_new=True))
    bb_std: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands standard deviation (1.5-3.0): ",
            prompt_on_new=True))
    macd_fast: int = Field(
        default=12,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period (8-16): ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period (20-30): ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period (7-12): ",
            prompt_on_new=True))
    stoch_rsi_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Stochastic RSI length (10-20): ",
            prompt_on_new=True))
    stoch_rsi_k: int = Field(
        default=3,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Stochastic RSI K period (3-5): ",
            prompt_on_new=True))
    stoch_rsi_d: int = Field(
        default=3,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Stochastic RSI D period (3-5): ",
            prompt_on_new=True))
    ichimoku_tenkan: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Ichimoku Tenkan-sen period (7-12): ",
            prompt_on_new=True))
    ichimoku_kijun: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Ichimoku Kijun-sen period (22-30): ",
            prompt_on_new=True))
    ichimoku_senkou_b: int = Field(
        default=52,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Ichimoku Senkou Span B period (44-60): ",
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

    def calculate_bollinger_bands(self, df, length=20, std=2.0):
        """Calculate Bollinger Bands for volatility-based reversals."""
        bb = df.ta.bbands(length=length, std=std)
        return bb
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD for trend momentum shifts."""
        macd = df.ta.macd(fast=fast, slow=slow, signal=signal)
        return macd
    
    def calculate_stoch_rsi(self, df, length=14, k=3, d=3):
        """Calculate Stochastic RSI for improved overbought/oversold detection."""
        stoch_rsi = df.ta.stochrsi(length=length, k=k, d=d)
        return stoch_rsi
    
    def calculate_ichimoku(self, df, tenkan=9, kijun=26, senkou_b=52):
        """Calculate Ichimoku Cloud for dynamic support/resistance."""
        ichimoku = df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou_b=senkou_b)
        return ichimoku
    
    def calculate_reversal_confidence(self, df, idx):
        """
        Calculate a confidence score for reversal signals.
        Returns a score between 0 and 1, with higher values indicating stronger reversal signals.
        """
        confidence = 0.0
        signal_count = 0
        
        # Get current values
        current_row = df.iloc[idx]
        
        # === Price Action Signals ===
        
        # Candlestick reversal patterns
        if idx > 0:
            prev_row = df.iloc[idx-1]
            
            # Bullish engulfing (for long)
            if (current_row['open'] < prev_row['close'] and 
                current_row['close'] > prev_row['open'] and
                current_row['signal'] == 1):
                confidence += 0.15
                signal_count += 1
            
            # Bearish engulfing (for short)
            if (current_row['open'] > prev_row['close'] and 
                current_row['close'] < prev_row['open'] and
                current_row['signal'] == -1):
                confidence += 0.15
                signal_count += 1
        
        # === Indicator Signals ===
        
        # Bollinger Band signals
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            # Bullish: Price touching lower band with positive momentum
            if (current_row['low'] <= current_row['BBL_20_2.0'] * 1.01 and 
                current_row['close'] > current_row['open'] and
                current_row['signal'] == 1):
                confidence += 0.12
                signal_count += 1
            
            # Bearish: Price touching upper band with negative momentum
            if (current_row['high'] >= current_row['BBU_20_2.0'] * 0.99 and 
                current_row['close'] < current_row['open'] and
                current_row['signal'] == -1):
                confidence += 0.12
                signal_count += 1
        
        # MACD signals
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            # Bullish: MACD crossing above signal line
            if (current_row['MACD_12_26_9'] > current_row['MACDs_12_26_9'] and
                df.iloc[idx-1]['MACD_12_26_9'] <= df.iloc[idx-1]['MACDs_12_26_9'] and
                current_row['signal'] == 1):
                confidence += 0.13
                signal_count += 1
            
            # Bearish: MACD crossing below signal line
            if (current_row['MACD_12_26_9'] < current_row['MACDs_12_26_9'] and
                df.iloc[idx-1]['MACD_12_26_9'] >= df.iloc[idx-1]['MACDs_12_26_9'] and
                current_row['signal'] == -1):
                confidence += 0.13
                signal_count += 1
        
        # Stochastic RSI signals
        if 'STOCHRSIk_14_14_3_3' in df.columns and 'STOCHRSId_14_14_3_3' in df.columns:
            # Bullish: StochRSI crossing above from oversold
            if (current_row['STOCHRSIk_14_14_3_3'] > current_row['STOCHRSId_14_14_3_3'] and
                current_row['STOCHRSIk_14_14_3_3'] < 0.3 and
                current_row['signal'] == 1):
                confidence += 0.14
                signal_count += 1
            
            # Bearish: StochRSI crossing below from overbought
            if (current_row['STOCHRSIk_14_14_3_3'] < current_row['STOCHRSId_14_14_3_3'] and
                current_row['STOCHRSIk_14_14_3_3'] > 0.7 and
                current_row['signal'] == -1):
                confidence += 0.14
                signal_count += 1
        
        # Ichimoku signals
        if 'ISA_9' in df.columns and 'ISB_26' in df.columns:
            # Bullish: Price crossing above Kumo (cloud)
            if (current_row['close'] > current_row['ISA_9'] and
                current_row['close'] > current_row['ISB_26'] and
                current_row['signal'] == 1):
                confidence += 0.11
                signal_count += 1
            
            # Bearish: Price crossing below Kumo (cloud)
            if (current_row['close'] < current_row['ISA_9'] and
                current_row['close'] < current_row['ISB_26'] and
                current_row['signal'] == -1):
                confidence += 0.11
                signal_count += 1
        
        # Fibonacci level signals
        if 'fib_PP' in df.columns:
            # Bullish: Price bouncing off support
            if (current_row['low'] <= current_row['fib_S1'] and
                current_row['close'] > current_row['fib_S1'] and
                current_row['signal'] == 1):
                confidence += 0.15
                signal_count += 1
            
            # Bearish: Price rejecting resistance
            if (current_row['high'] >= current_row['fib_R1'] and
                current_row['close'] < current_row['fib_R1'] and
                current_row['signal'] == -1):
                confidence += 0.15
                signal_count += 1
        
        # Volume confirmation
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(window=20).mean()
            if idx >= 20:
                # Strong volume on reversal
                if current_row['volume'] > volume_ma.iloc[idx] * 1.5:
                    confidence += 0.1
                    signal_count += 1
        
        # Normalize confidence score if we have signals
        if signal_count > 0:
            # Adjust confidence based on signal count (more signals = higher confidence)
            confidence_multiplier = min(1.0, 0.7 + (signal_count / 10))
            return min(1.0, confidence * confidence_multiplier)
        
        return 0.0

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
            
        # Calculate new enhanced indicators
        bb = self.calculate_bollinger_bands(df, length=self.config.bb_length, std=self.config.bb_std)
        macd = self.calculate_macd(df, fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal)
        stoch_rsi = self.calculate_stoch_rsi(df, length=self.config.stoch_rsi_length, k=self.config.stoch_rsi_k, d=self.config.stoch_rsi_d)
        ichimoku = self.calculate_ichimoku(df, tenkan=self.config.ichimoku_tenkan, kijun=self.config.ichimoku_kijun, senkou_b=self.config.ichimoku_senkou_b)
        
        # Add indicators to dataframe
        for col in bb.columns:
            df[col] = bb[col]
        
        for col in macd.columns:
            df[col] = macd[col]
            
        for col in stoch_rsi.columns:
            df[col] = stoch_rsi[col]
            
        for col in ichimoku.columns:
            df[col] = ichimoku[col]

        # Signal generation
        df['signal'] = 0
        
        # ===== ENHANCED REVERSAL SIGNALS =====
        
        # RSI divergence calculation
        rsi_higher_high = (rsi > rsi.shift(1)) & (rsi.shift(1) > rsi.shift(2))
        rsi_lower_low = (rsi < rsi.shift(1)) & (rsi.shift(1) < rsi.shift(2))
        price_higher_high = (close > close.shift(1)) & (close.shift(1) > close.shift(2))
        price_lower_low = (close < close.shift(1)) & (close.shift(1) < close.shift(2))
        
        # Bearish divergence: Price makes higher high but RSI makes lower high
        bearish_divergence = price_higher_high & ~rsi_higher_high & (rsi.shift(1) > 70)
        
        # Bullish divergence: Price makes lower low but RSI makes higher low
        bullish_divergence = price_lower_low & ~rsi_lower_low & (rsi.shift(1) < 30)
        
        # Bollinger Band reversal signals
        bb_upper = df['BBU_20_2.0'] if 'BBU_20_2.0' in df.columns else None
        bb_lower = df['BBL_20_2.0'] if 'BBL_20_2.0' in df.columns else None
        bb_middle = df['BBM_20_2.0'] if 'BBM_20_2.0' in df.columns else None
        
        if bb_upper is not None and bb_lower is not None and bb_middle is not None:
            # Bollinger Band squeeze (volatility contraction)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_squeeze = bb_width < bb_width.rolling(window=20).quantile(0.2)
            
            # Bollinger Band breakout after squeeze
            bb_breakout_up = (close > bb_upper) & bb_squeeze.shift(1) & (close.shift(1) <= bb_upper.shift(1))
            bb_breakout_down = (close < bb_lower) & bb_squeeze.shift(1) & (close.shift(1) >= bb_lower.shift(1))
            
            # Bollinger Band mean reversion
            bb_touch_upper = (high >= bb_upper) & (close < bb_upper)
            bb_touch_lower = (low <= bb_lower) & (close > bb_lower)
        else:
            bb_squeeze = pd.Series(False, index=df.index)
            bb_breakout_up = pd.Series(False, index=df.index)
            bb_breakout_down = pd.Series(False, index=df.index)
            bb_touch_upper = pd.Series(False, index=df.index)
            bb_touch_lower = pd.Series(False, index=df.index)
        
        # MACD signals
        macd_line = df['MACD_12_26_9'] if 'MACD_12_26_9' in df.columns else None
        macd_signal = df['MACDs_12_26_9'] if 'MACDs_12_26_9' in df.columns else None
        
        if macd_line is not None and macd_signal is not None:
            # MACD crossovers
            macd_cross_above = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
            macd_cross_below = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
            
            # MACD divergence
            macd_higher_high = (macd_line > macd_line.shift(1)) & (macd_line.shift(1) > macd_line.shift(2))
            macd_lower_low = (macd_line < macd_line.shift(1)) & (macd_line.shift(1) < macd_line.shift(2))
            
            macd_bearish_div = price_higher_high & ~macd_higher_high & (market_trend == 1)
            macd_bullish_div = price_lower_low & ~macd_lower_low & (market_trend == -1)
        else:
            macd_cross_above = pd.Series(False, index=df.index)
            macd_cross_below = pd.Series(False, index=df.index)
            macd_bearish_div = pd.Series(False, index=df.index)
            macd_bullish_div = pd.Series(False, index=df.index)
        
        # Stochastic RSI signals
        stoch_rsi_k = df['STOCHRSIk_14_14_3_3'] if 'STOCHRSIk_14_14_3_3' in df.columns else None
        stoch_rsi_d = df['STOCHRSId_14_14_3_3'] if 'STOCHRSId_14_14_3_3' in df.columns else None
        
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            # StochRSI crossovers
            stoch_rsi_cross_above = (stoch_rsi_k > stoch_rsi_d) & (stoch_rsi_k.shift(1) <= stoch_rsi_d.shift(1))
            stoch_rsi_cross_below = (stoch_rsi_k < stoch_rsi_d) & (stoch_rsi_k.shift(1) >= stoch_rsi_d.shift(1))
            
            # StochRSI overbought/oversold
            stoch_rsi_oversold = (stoch_rsi_k < 0.2) & (stoch_rsi_k > stoch_rsi_k.shift(1))
            stoch_rsi_overbought = (stoch_rsi_k > 0.8) & (stoch_rsi_k < stoch_rsi_k.shift(1))
        else:
            stoch_rsi_cross_above = pd.Series(False, index=df.index)
            stoch_rsi_cross_below = pd.Series(False, index=df.index)
            stoch_rsi_oversold = pd.Series(False, index=df.index)
            stoch_rsi_overbought = pd.Series(False, index=df.index)
        
        # Ichimoku signals
        tenkan = df['ITS_9'] if 'ITS_9' in df.columns else None
        kijun = df['IKS_26'] if 'IKS_26' in df.columns else None
        senkou_a = df['ISA_9'] if 'ISA_9' in df.columns else None
        senkou_b = df['ISB_26'] if 'ISB_26' in df.columns else None
        
        if tenkan is not None and kijun is not None and senkou_a is not None and senkou_b is not None:
            # TK Cross (Tenkan-Kijun cross)
            tk_cross_bull = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
            tk_cross_bear = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
            
            # Kumo breakouts
            kumo_breakout_bull = (close > senkou_a) & (close > senkou_b) & ((close.shift(1) <= senkou_a.shift(1)) | (close.shift(1) <= senkou_b.shift(1)))
            kumo_breakout_bear = (close < senkou_a) & (close < senkou_b) & ((close.shift(1) >= senkou_a.shift(1)) | (close.shift(1) >= senkou_b.shift(1)))
        else:
            tk_cross_bull = pd.Series(False, index=df.index)
            tk_cross_bear = pd.Series(False, index=df.index)
            kumo_breakout_bull = pd.Series(False, index=df.index)
            kumo_breakout_bear = pd.Series(False, index=df.index)
        
        # Combine reversal signals for long entries
        reversal_long = (
            (bullish_divergence | macd_bullish_div) |  # Divergence signals
            (bb_touch_lower & stoch_rsi_oversold) |    # BB + StochRSI confirmation
            (macd_cross_above & stoch_rsi_cross_above & (market_trend == -1)) |  # Multiple indicator confirmation
            (tk_cross_bull & kumo_breakout_bull) |     # Ichimoku confirmation
            (bb_breakout_up & volume_surge)            # Volatility breakout with volume
        )
        
        # Combine reversal signals for short entries
        reversal_short = (
            (bearish_divergence | macd_bearish_div) |  # Divergence signals
            (bb_touch_upper & stoch_rsi_overbought) |  # BB + StochRSI confirmation
            (macd_cross_below & stoch_rsi_cross_below & (market_trend == 1)) |  # Multiple indicator confirmation
            (tk_cross_bear & kumo_breakout_bear) |     # Ichimoku confirmation
            (bb_breakout_down & volume_surge)          # Volatility breakout with volume
        )
        
        # Apply decimation to reduce noise
        reversal_long = self.decimate(reversal_long)
        reversal_short = self.decimate(reversal_short)

        df.loc[reversal_long, 'signal'] = 1
        df.loc[reversal_short, 'signal'] = -1
        
        # Calculate confidence scores for each signal
        df['confidence'] = 0.0
        for i in range(len(df)):
            if df['signal'].iloc[i] != 0:
                df['confidence'].iloc[i] = self.calculate_reversal_confidence(df, i)
        
        # Add signal type for analysis
        df['signal_type'] = 'none'
        
        # Bullish signal types
        df.loc[bullish_divergence & reversal_long, 'signal_type'] = 'bullish_divergence'
        df.loc[macd_bullish_div & reversal_long, 'signal_type'] = 'macd_bullish_divergence'
        df.loc[bb_touch_lower & stoch_rsi_oversold & reversal_long, 'signal_type'] = 'bb_stochrsi_oversold'
        df.loc[macd_cross_above & stoch_rsi_cross_above & (market_trend == -1) & reversal_long, 'signal_type'] = 'multi_indicator_bullish'
        df.loc[tk_cross_bull & kumo_breakout_bull & reversal_long, 'signal_type'] = 'ichimoku_bullish'
        df.loc[bb_breakout_up & volume_surge & reversal_long, 'signal_type'] = 'bb_breakout_bullish'
        
        # Bearish signal types
        df.loc[bearish_divergence & reversal_short, 'signal_type'] = 'bearish_divergence'
        df.loc[macd_bearish_div & reversal_short, 'signal_type'] = 'macd_bearish_divergence'
        df.loc[bb_touch_upper & stoch_rsi_overbought & reversal_short, 'signal_type'] = 'bb_stochrsi_overbought'
        df.loc[macd_cross_below & stoch_rsi_cross_below & (market_trend == 1) & reversal_short, 'signal_type'] = 'multi_indicator_bearish'
        df.loc[tk_cross_bear & kumo_breakout_bear & reversal_short, 'signal_type'] = 'ichimoku_bearish'
        df.loc[bb_breakout_down & volume_surge & reversal_short, 'signal_type'] = 'bb_breakout_bearish'

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["confidence"] = df["confidence"].iloc[-1]
        self.processed_data["features"] = df
        
        #