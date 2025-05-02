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
import json
import os
from math import sqrt, exp  # Added exp for CVIX calculation


class ZLEMAConfig(DirectionalTradingControllerConfigBase):
    controller_name = "zlemogg"
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
    # --- Fractal Parameters ---
    fractal_window: int = Field(
        default=5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the fractal window size (2-10): ",
            prompt_on_new=True))
    fractal_lookback_period: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the lookback period for fractal confirmation (5-20): ",
            prompt_on_new=True))
    # --- End Fractal Parameters ---
    
    # --- CVIX Parameters ---
    cvix_period: int = Field(
        default=30,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the CVIX calculation period (10-90): ",
            prompt_on_new=True))
    cvix_threshold_high: float = Field(
        default=80.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the high CVIX threshold (60-90): ",
            prompt_on_new=True))
    cvix_threshold_low: float = Field(
        default=20.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the low CVIX threshold (10-40): ",
            prompt_on_new=True))
    # --- End CVIX Parameters ---

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


# Added helper to detect fractal patterns
def detect_fractals(high_prices: pd.Series, low_prices: pd.Series, window: int = 5):
    """Detect bullish and bearish fractals in price series."""
    bullish_fractals = []
    bearish_fractals = []
    n = len(high_prices)
    for i in range(window, n - window):
        # Bullish fractal: current low is lower than surrounding lows
        if all(low_prices.iloc[i] < low_prices.iloc[i - j] for j in range(1, window + 1)) and \
           all(low_prices.iloc[i] < low_prices.iloc[i + j] for j in range(1, window + 1)):
            bullish_fractals.append((i, low_prices.iloc[i]))
        # Bearish fractal: current high is higher than surrounding highs
        if all(high_prices.iloc[i] > high_prices.iloc[i - j] for j in range(1, window + 1)) and \
           all(high_prices.iloc[i] > high_prices.iloc[i + j] for j in range(1, window + 1)):
            bearish_fractals.append((i, high_prices.iloc[i]))
    return bullish_fractals, bearish_fractals


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
        self.controller_status_message = ""

    def initialize_controller_status_message(self) -> str:
        """Initialize the controller status message with basic information"""
        status_msg = (
            f"ZLEMA Controller Status:"
            f"\n  Connector: {self.config.candles_connector}"
            f"\n  Trading Pair: {self.config.candles_trading_pair}"
            f"\n  Interval: {self.config.interval}"
            f"\n  Price Source: {self.config.source}"
            f"\n  Kalman Filter: Enabled" 
            f"\n  Fast/Medium/Slow Periods: {self.config.period_fast}/{self.config.period_medium}/{self.config.period_slow}"
            f"\n  Fractal Confirmation: Enabled (Window: {self.config.fractal_window}, Lookback: {self.config.fractal_lookback_period})"
            f"\n  CVIX Integration: Enabled"
            f"\n  CVIX Settings: Period={self.config.cvix_period}, High={self.config.cvix_threshold_high}, Low={self.config.cvix_threshold_low}"
        )
        self.controller_status_message = status_msg
        return status_msg

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
                value1 = 0.2 * (price - prev_price) + 0.8 * value1
                value2 = 0.1 * tr.iloc[i] + 0.8 * value2
                
                # Calculate lambda and alpha
                if value2 != 0:
                    lambda_val = abs(value1 / value2)
                    alpha = (-pow(lambda_val, 2) + sqrt(pow(lambda_val, 4) + 16 * pow(lambda_val, 2))) / 8
                else:
                    alpha = 0.1  # Default value if division by zero
                
                # Apply filter
                value3 = alpha * price + (1 - alpha) * value3
            else:
                # Initialize with first value
                value3 = price
            
            result.iloc[i] = value3
        
        return result
        
    def calculate_cvix(self, close_prices: pd.Series, period: int = 30) -> pd.Series:
        """
        Calculate the Crypto Volatility Index (CVIX) similar to the CBOE VIX
        
        Args:
            close_prices: Series of closing prices
            period: Lookback period for calculation (default: 30)
            
        Returns:
            Series containing CVIX values normalized to 0-100 scale
        """
        # Ensure we have enough data
        if len(close_prices) < period + 1:
            return pd.Series(index=close_prices.index, data=np.nan)
            
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        # Create result series
        cvix = pd.Series(index=close_prices.index, data=np.nan)
        
        # Calculate rolling volatility (annualized)
        # Use log returns for better statistical properties
        log_returns = np.log(1 + returns)
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize (based on interval - assuming daily data by default)
        # For different timeframes, should adjust this multiplier
        annualized_vol = rolling_std * np.sqrt(365)  
        
        # Convert to CVIX scale (0-100)
        # Similar to how VIX is calculated but simplified
        # VIX = 100 * σ, where σ is the annualized volatility
        raw_cvix = 100 * annualized_vol
        
        # Apply normalization and scaling similar to VIX methodology
        # Normalize to 0-100 range where:
        # - Values > 80 indicate extreme fear/volatility
        # - Values < 20 indicate extreme complacency/low volatility
        min_val = raw_cvix.min()
        max_val = raw_cvix.max()
        
        if max_val > min_val:
            normalized_cvix = 100 * (raw_cvix - min_val) / (max_val - min_val)
            
            # Apply sigmoid-like transformation to emphasize extreme values
            cvix = normalized_cvix.apply(lambda x: 100 / (1 + exp(-0.1 * (x - 50))))
        else:
            cvix = raw_cvix  # Fallback if all values are the same
        
        return cvix

    def analyze_kalman_impact(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the impact of Kalman filter on trading signals
        """
        if 'raw_source' not in df or 'filtered_source' not in df:
            return {"error": "Raw and filtered source data not available"}
            
        # Calculate signals with and without Kalman filter
        raw_ma1 = self.zlema(df['raw_source'], self.config.period_fast)
        raw_ma2 = self.zlema(df['raw_source'], self.config.period_medium)
        
        # Generate raw signals (without Kalman)
        raw_buy_signal = (raw_ma1 > raw_ma2) & (raw_ma1.shift(1) <= raw_ma2.shift(1))
        raw_sell_signal = (raw_ma1 < raw_ma2) & (raw_ma1.shift(1) >= raw_ma2.shift(1))
        
        # Compare with filtered signals
        missed_buys = raw_buy_signal & ~df['buy_signal']
        missed_sells = raw_sell_signal & ~df['sell_signal']
        extra_buys = ~raw_buy_signal & df['buy_signal']
        extra_sells = ~raw_sell_signal & df['sell_signal']
        
        # Signal timing analysis
        signal_delays = {
            'buy_delays': [],
            'sell_delays': []
        }
        
        for i in range(len(df)):
            if raw_buy_signal.iloc[i]:
                # Look ahead to find if filtered signal triggers later
                for j in range(i, min(i + 20, len(df))):
                    if df['buy_signal'].iloc[j]:
                        signal_delays['buy_delays'].append(j - i)
                        break
                        
            if raw_sell_signal.iloc[i]:
                # Look ahead to find if filtered signal triggers later
                for j in range(i, min(i + 20, len(df))):
                    if df['sell_signal'].iloc[j]:
                        signal_delays['sell_delays'].append(j - i)
                        break
        
        # Prepare analytics
        results = {
            "total_raw_buy_signals": raw_buy_signal.sum(),
            "total_raw_sell_signals": raw_sell_signal.sum(),
            "total_filtered_buy_signals": df['buy_signal'].sum(),
            "total_filtered_sell_signals": df['sell_signal'].sum(),
            "missed_buy_signals": missed_buys.sum(),
            "missed_sell_signals": missed_sells.sum(),
            "extra_buy_signals": extra_buys.sum(),
            "extra_sell_signals": extra_sells.sum(),
            "avg_buy_delay": np.mean(signal_delays['buy_delays']) if signal_delays['buy_delays'] else 0,
            "avg_sell_delay": np.mean(signal_delays['sell_delays']) if signal_delays['sell_delays'] else 0,
            "max_price_smoothing": (df['filtered_source'] - df['raw_source']).abs().max(),
            "avg_price_smoothing": (df['filtered_source'] - df['raw_source']).abs().mean()
        }
        
        return results

    def zlema(self, src: pd.Series, period: int) -> pd.Series:
        """
        Zero Lag Exponential Moving Average implementation
        https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
        """
        # Calculate lag
        lag = int((period - 1) / 2)
        
        # Ensure src has enough data for the shift
        if len(src) <= lag:
            # Return a series of NaNs or handle as appropriate
            return pd.Series(index=src.index, dtype=float)

        # Create ema_data
        ema_data = src + (src - src.shift(lag))
        
        # Calculate EMA of ema_data, handle potential NaNs from ema_data
        return ema_data.ewm(span=period, adjust=False).mean()

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        # Ensure dataframe is not empty and has required columns
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            self.logger().warning("Candles DataFrame is empty or missing required columns (OHLC). Skipping update.")
            # Ensure processed_data defaults are set
            self.processed_data["signal"] = 0
            self.processed_data["features"] = pd.DataFrame()
            self.processed_data["kalman_impact"] = {"error": "No candle data"}
            return

        # --- Pre-computation ---
        # Get price source
        src = self.get_price_source(df)

        # Apply Kalman filter if source has enough data
        filtered_src = self.kalman_filter(src) if len(src) > 1 else src.copy()

        # Calculate ZLEMAs using filtered source
        ma1 = self.zlema(filtered_src, self.config.period_fast)
        ma2 = self.zlema(filtered_src, self.config.period_medium)
        ma3 = self.zlema(filtered_src, self.config.period_slow)

        # Calculate ATR for volatility context (using raw OHLC)
        # Ensure enough data for ATR calculation, use default length 14
        atr_length = 14
        if len(df) > atr_length:
             df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
        else:
             df['atr'] = pd.NA # Not enough data to calculate

        # Calculate CVIX - this is now mandatory
        df['cvix'] = self.calculate_cvix(df['close'], self.config.cvix_period)

        # --- Store calculated values in DataFrame ---
        df['zlema_fast'] = ma1
        df['zlema_medium'] = ma2
        df['zlema_slow'] = ma3
        df['raw_source'] = src
        df['filtered_source'] = filtered_src
        df['filter_diff'] = filtered_src - src

        # Fractal detection
        bullish_fractals, bearish_fractals = detect_fractals(df['high'], df['low'], window=self.config.fractal_window)

        # --- Generate signals based on ZLEMA cross with fractal and CVIX confirmation ---
        if len(df) > 1:
            # Initial ZLEMA cross conditions
            buy_condition = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
            sell_condition = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))
            
            # Confirm fractal for latest bar within lookback period
            latest_pos = len(df) - 1
            if buy_condition.iat[latest_pos]:
                if not any(idx >= latest_pos - self.config.fractal_lookback_period for idx, _ in bullish_fractals):
                    buy_condition.iloc[-1] = False
            if sell_condition.iat[latest_pos]:
                if not any(idx >= latest_pos - self.config.fractal_lookback_period for idx, _ in bearish_fractals):
                    sell_condition.iloc[-1] = False
                    
            # Apply CVIX filter as mandatory part of strategy
            if 'cvix' in df and not df['cvix'].isna().all():
                # For buy signals - only allow when CVIX is below high threshold (lower volatility)
                if buy_condition.iat[latest_pos]:
                    current_cvix = df['cvix'].iloc[-1]
                    if current_cvix >= self.config.cvix_threshold_high:
                        buy_condition.iloc[-1] = False  # Too volatile, filter out buy
                        
                # For sell signals - strengthen signals when CVIX is above high threshold (high volatility)
                # but filter out sells when CVIX is below low threshold (stable market)
                if sell_condition.iat[latest_pos]:
                    current_cvix = df['cvix'].iloc[-1]
                    if current_cvix <= self.config.cvix_threshold_low:
                        sell_condition.iloc[-1] = False  # Too stable, filter out sell
            else:
                # If CVIX is not available (early bars), don't generate signals
                buy_condition.iloc[-1] = False
                sell_condition.iloc[-1] = False
                self.logger().warning("CVIX data not available - signals suppressed until CVIX calculation stabilizes")
            
            df['buy_signal'] = buy_condition
            df['sell_signal'] = sell_condition
        else:
            df['buy_signal'] = False
            df['sell_signal'] = False

        # Set signal values
        df['signal'] = 0
        df.loc[df['buy_signal'], 'signal'] = 1
        df.loc[df['sell_signal'], 'signal'] = -1

        # Update processed data only
        self.processed_data.update(df.iloc[-1].to_dict())
        self.processed_data["features"] = df
        
    # No additional analytics or logging beyond processed_data update
    
    def log_kalman_metrics(self, kalman_impact: Dict):  # kept for interface compatibility but no-op
        return

    def format_status(self) -> str:
        """
        Format status message to include strategy info
        """
        if not hasattr(self, 'controller_status_message'):
            return "ZLEMA Controller: No data available yet."
            
        status = self.controller_status_message
        
        # Add latest CVIX value if available
        if hasattr(self, 'processed_data') and self.processed_data and 'cvix' in self.processed_data:
            cvix_value = self.processed_data.get('cvix', 'N/A')
            if cvix_value != 'N/A' and not pd.isna(cvix_value):
                cvix_str = f"{cvix_value:.2f}"
                status += f"\n  Current CVIX: {cvix_str}"
                
                # Interpret CVIX level
                if cvix_value >= self.config.cvix_threshold_high:
                    status += " (High Volatility - Buy signals suppressed)"
                elif cvix_value <= self.config.cvix_threshold_low:
                    status += " (Low Volatility - Sell signals suppressed)"
                else:
                    status += " (Normal Volatility - Regular trading)"
                    
        return status