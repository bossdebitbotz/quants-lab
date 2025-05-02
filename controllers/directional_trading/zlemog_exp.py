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
import logging
import json
import os
from math import sqrt


class ZLEMAConfig(DirectionalTradingControllerConfigBase):
    controller_name = "zlemog_exp"
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

    # --- Parameters for "Breaking the Trend" Paper EMA ---
    paper_ema_enabled: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enable calculation of the 'Breaking the Trend' paper EMA signal? (True/False)",
            prompt_on_new=True)
    )
    paper_ema_eta: float = Field(
        default=1/112.0, # Optimal value from paper (for daily data)
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the eta parameter for the paper EMA (e.g., 1/112): ",
            prompt_on_new=True)
    )
    volatility_ema_period: int = Field(
        default=40, # Period from paper (for daily data)
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the period for volatility EMA calculation (e.g., 40): ",
            prompt_on_new=True)
    )
    # --- End Parameters ---

    # --- Filter using Paper EMA ---
    paper_ema_trend_threshold: float = Field(
        default=0.0, # Set to > 0 to require stronger trend confirmation
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the trend threshold for Paper EMA filtering (e.g., 0.0): ",
            prompt_on_new=True)
    )
    # --- End Filter ---

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
        self.controller_status_message = ""
        self._setup_tuning_logger()
        self._log_initial_config()

    def _setup_tuning_logger(self):
        """Sets up a dedicated logger for tuning data."""
        self.tuning_log_file = f"{self.config.controller_name}_tuning.log" # Use controller name
        self.tuning_logger = logging.getLogger(f"{self.__class__.__name__}_Tuning")
        self.tuning_logger.setLevel(logging.INFO)
        # Avoid adding duplicate handlers if instance is recreated or multiple instances run
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(self.tuning_log_file) for h in self.tuning_logger.handlers):
            fh = logging.FileHandler(self.tuning_log_file)
            # Use JSON format for easy parsing
            formatter = logging.Formatter('{"timestamp": "%(asctime)s.%(msecs)03d", "level": "%(levelname)s", "data": %(message)s}', datefmt='%Y-%m-%dT%H:%M:%S')
            fh.setFormatter(formatter)
            self.tuning_logger.addHandler(fh)
        # Prevent logs from propagating to the root logger
        self.tuning_logger.propagate = False


    def _log_initial_config(self):
        """Logs the initial configuration parameters."""
        config_data = {
            "event": "config_loaded",
            "controller_name": self.config.controller_name,
            "connector": self.config.candles_connector,
            "trading_pair": self.config.candles_trading_pair,
            "interval": self.config.interval,
            "source": self.config.source,
            "period_fast": self.config.period_fast,
            "period_medium": self.config.period_medium,
            "period_slow": self.config.period_slow,
            "max_records": self.max_records,
            # --- Add Paper EMA Config ---
            "paper_ema_enabled": self.config.paper_ema_enabled,
            "paper_ema_eta": self.config.paper_ema_eta,
            "volatility_ema_period": self.config.volatility_ema_period,
            # --- End Paper EMA Config ---
        }
        # Use default=str for safety, although config types should be fine
        self.tuning_logger.info(json.dumps(config_data, default=str))

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
            # --- Add Paper EMA Info ---
            f"\n  Paper EMA (η={self.config.paper_ema_eta:.4f}, Vol Period={self.config.volatility_ema_period}): {'Enabled' if self.config.paper_ema_enabled else 'Disabled'}"
            # --- End Paper EMA Info ---
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
                    alpha = (-pow(lambda_val, 2) + np.sqrt(pow(lambda_val, 4) + 16 * pow(lambda_val, 2))) / 8
                else:
                    alpha = 0.1  # Default value if division by zero
                
                # Apply filter
                value3 = alpha * price + (1 - alpha) * value3
            else:
                # Initialize with first value
                value3 = price
            
            result.iloc[i] = value3
        
        return result

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


        # --- Store calculated values in DataFrame ---
        df['zlema_fast'] = ma1
        df['zlema_medium'] = ma2
        df['zlema_slow'] = ma3
        df['raw_source'] = src
        df['filtered_source'] = filtered_src
        df['filter_diff'] = filtered_src - src

        # --- Generate signals based on crosses ---
        # Need at least 2 rows to compare shift(1)
        if len(df) > 1:
            # Crossover (buy signal) - Apply Paper EMA trend filter if enabled
            buy_condition = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
            if self.config.paper_ema_enabled:
                buy_condition &= (df['paper_ema'] > self.config.paper_ema_trend_threshold)
            df['buy_signal'] = buy_condition

            # Crossunder (sell signal) - Apply Paper EMA trend filter if enabled
            sell_condition = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))
            if self.config.paper_ema_enabled:
                sell_condition &= (df['paper_ema'] < -self.config.paper_ema_trend_threshold)
            df['sell_signal'] = sell_condition
        else:
            df['buy_signal'] = False
            df['sell_signal'] = False

        # Set signal values
        df['signal'] = 0
        df.loc[df['buy_signal'], 'signal'] = 1
        df.loc[df['sell_signal'], 'signal'] = -1

        # --- Calculate "Breaking the Trend" Paper EMA Signal (if enabled) ---
        latest_paper_ema = None
        if self.config.paper_ema_enabled and len(df) > 1:
            # 1. Calculate returns (using filtered source like ZLEMA)
            df['returns'] = df['filtered_source'].pct_change()

            # 2. Calculate volatility estimate (EMA of squared returns)
            # Need enough data for volatility period
            if len(df) > self.config.volatility_ema_period:
                # Use pandas ewm directly for variance
                variance_ema = (df['returns']**2).ewm(span=self.config.volatility_ema_period, adjust=False).mean()
                # Avoid zero or NaN sigma, replace with small number or previous valid? Using fillna with backfill first
                df['volatility'] = np.sqrt(variance_ema).fillna(method='bfill').fillna(1e-9) # Estimate sigma (std dev)
            else:
                df['volatility'] = pd.NA # Not enough data

            # 3. Calculate the normalized EMA signal
            # Formula: EMA_{t+1} = (1-η) * EMA_{t} + sqrt(η) * (r_{t+1} / σ_{t})
            # Requires iterative calculation due to dependence on previous sigma (σ_t)
            eta = self.config.paper_ema_eta
            sqrt_eta = sqrt(eta)
            paper_ema_values = np.full(len(df), np.nan)

            if 'volatility' in df and not df['volatility'].isnull().all() and not df['returns'].isnull().all():
                # Initialize first value (can refine this initialization)
                first_valid_idx = df['returns'].first_valid_index()
                if first_valid_idx is not None and first_valid_idx < len(df) -1 :
                    # Start calculation from the second available return
                    paper_ema_values[first_valid_idx + 1] = 0 # Initial EMA value set to 0

                    for i in range(df.index.get_loc(first_valid_idx) + 2, len(df)):
                        prev_ema = paper_ema_values[i-1]
                        current_return = df['returns'].iloc[i]
                        prev_sigma = df['volatility'].iloc[i-1] # Use sigma from t

                        # Check for valid data before calculation
                        if pd.notna(prev_ema) and pd.notna(current_return) and pd.notna(prev_sigma) and prev_sigma > 1e-9:
                            normalized_return = sqrt_eta * (current_return / prev_sigma)
                            paper_ema_values[i] = (1 - eta) * prev_ema + normalized_return
                        else:
                            # Propagate previous EMA if data is invalid
                            paper_ema_values[i] = prev_ema

            df['paper_ema'] = paper_ema_values
            latest_paper_ema = df['paper_ema'].iloc[-1] if not df['paper_ema'].empty and pd.notna(df['paper_ema'].iloc[-1]) else None
        else:
            df['paper_ema'] = pd.NA # Disabled or not enough data
            df['returns'] = pd.NA
            df['volatility'] = pd.NA


        # --- Analyze Kalman filter impact ---
        # Ensure required columns exist for analysis
        if 'raw_source' in df and 'filtered_source' in df and 'buy_signal' in df and 'sell_signal' in df and len(df) > 1:
             kalman_impact = self.analyze_kalman_impact(df)
        else:
             kalman_impact = {"error": "Insufficient data or missing columns for Kalman analysis"}


        # --- Update processed data dictionary ---
        self.processed_data["signal"] = df["signal"].iloc[-1] if not df["signal"].empty else 0
        self.processed_data["features"] = df
        self.processed_data["kalman_impact"] = kalman_impact
        # --- Add Paper EMA to processed data ---
        self.processed_data["paper_ema"] = latest_paper_ema
        # --- End Paper EMA ---

        # --- Logging for Tuning ---
        # 1. Log Kalman impact periodically
        kalman_log_data = {
            "event": "kalman_update",
            **kalman_impact # Unpack the dictionary
        }
        # Use default=str to handle potential numpy types etc.
        self.tuning_logger.info(json.dumps(kalman_log_data, default=str))

        # 2. Log signal events if they occur on the latest candle
        if not df.empty:
            latest_candle = df.iloc[-1]
            if latest_candle["signal"] != 0:
                signal_data = {
                    "event": "signal_generated",
                    "signal_type": "buy" if latest_candle["signal"] == 1 else "sell",
                    "candle_timestamp": latest_candle.name.isoformat() if isinstance(latest_candle.name, pd.Timestamp) else str(latest_candle.name),
                    "zlema_fast": latest_candle["zlema_fast"],
                    "zlema_medium": latest_candle["zlema_medium"],
                    "zlema_slow": latest_candle["zlema_slow"],
                    "raw_source": latest_candle["raw_source"],
                    "filtered_source": latest_candle["filtered_source"],
                    "atr": latest_candle.get("atr", None), # Include ATR if calculated
                    # --- Add Paper EMA to Signal Log ---
                    "paper_ema": latest_candle.get("paper_ema", None),
                    "volatility": latest_candle.get("volatility", None),
                    # --- End Paper EMA ---
                }
                self.tuning_logger.info(json.dumps(signal_data, default=str))


        # --- Status Updates & Warnings ---
        # Initialize controller status message (can be done earlier)
        self.initialize_controller_status_message()

        # Always print Kalman filter impact metrics in standard logs
        self.log_kalman_metrics(kalman_impact)

        # Log warning if Kalman filter is significantly reducing signals
        if "error" not in kalman_impact:
            total_raw_signals = kalman_impact.get("total_raw_buy_signals", 0) + kalman_impact.get("total_raw_sell_signals", 0)
            total_filtered_signals = kalman_impact.get("total_filtered_buy_signals", 0) + kalman_impact.get("total_filtered_sell_signals", 0)

            if total_raw_signals > 0 and (total_filtered_signals / total_raw_signals) < 0.7:
                 self.logger().warning(
                     f"Kalman filter may be inhibiting trading - only {total_filtered_signals}/{total_raw_signals} "
                     f"signals triggered (reducing signals by {(1 - total_filtered_signals / total_raw_signals) * 100:.1f}%)"
                 )

            if kalman_impact.get("avg_buy_delay", 0) > 2 or kalman_impact.get("avg_sell_delay", 0) > 2:
                 self.logger().warning(
                     f"Kalman filter is delaying signals - avg buy delay: {kalman_impact.get('avg_buy_delay', 0):.1f} candles, "
                     f"avg sell delay: {kalman_impact.get('avg_sell_delay', 0):.1f} candles"
                 )

    def log_kalman_metrics(self, kalman_impact: Dict):
        """
        Log Kalman filter impact metrics to the console
        """
        if "error" in kalman_impact:
            self.logger().info("Kalman filter impact: Unable to calculate metrics")
            return
            
        # Calculate signal reduction percentage
        total_raw = kalman_impact.get("total_raw_buy_signals", 0) + kalman_impact.get("total_raw_sell_signals", 0)
        total_filtered = kalman_impact.get("total_filtered_buy_signals", 0) + kalman_impact.get("total_filtered_sell_signals", 0)
        
        if total_raw > 0:
            signal_reduction = (1 - total_filtered / total_raw) * 100
        else:
            signal_reduction = 0
            
        # Format status message
        status_msg = (
            f"KALMAN FILTER STATUS:"
            f"\n  Signal Reduction: {signal_reduction:.1f}% "
            f"(Raw: {total_raw}, Filtered: {total_filtered})"
            f"\n  Missed Signals: Buy={kalman_impact.get('missed_buy_signals', 0)}, "
            f"Sell={kalman_impact.get('missed_sell_signals', 0)}"
            f"\n  Added Signals: Buy={kalman_impact.get('extra_buy_signals', 0)}, "
            f"Sell={kalman_impact.get('extra_sell_signals', 0)}"
            f"\n  Avg Signal Delay: Buy={kalman_impact.get('avg_buy_delay', 0):.1f} candles, "
            f"Sell={kalman_impact.get('avg_sell_delay', 0):.1f} candles"
            f"\n  Price Smoothing: Avg={kalman_impact.get('avg_price_smoothing', 0):.6f}, "
            f"Max={kalman_impact.get('max_price_smoothing', 0):.6f}"
        )
        
        # Log status message
        self.logger().info(status_msg)
        
        # Append to status message (this will be displayed in status updates)
        # Add paper EMA status if calculated
        paper_ema_status = f"\n  Paper EMA Value: {self.processed_data.get('paper_ema', 'N/A'):.4f}" if self.config.paper_ema_enabled and self.processed_data.get('paper_ema') is not None else ""
        
        if hasattr(self, 'controller_status_message'):
            # Reset base message and append current metrics
            base_status = self.initialize_controller_status_message() # Get the base static info
            self.controller_status_message = f"{base_status}\n{status_msg}{paper_ema_status}" 
        else:
            self.controller_status_message = f"{self.initialize_controller_status_message()}\n{status_msg}{paper_ema_status}" # Initialize if first time

    def format_status(self) -> str:
        """
        Format status message to include Kalman filter metrics
        """
        if not hasattr(self, 'controller_status_message'):
            return "ZLEMA Controller: No data available yet."
            
        return self.controller_status_message