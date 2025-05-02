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
    controller_name = "zlem"
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
    # Kalman filter is always enabled
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
    # ZLEMA crosses are always used for signals instead of alignment
    # --- SOTT and DeMark Configuration ---
    # Both SOTT and DeMark confirmations are always enabled by default
    # DeMark parameters
    demark_lookback_period: int = Field(
        default=1,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the DeMark lookback period (e.g., 1 for previous bar): ",
            prompt_on_new=True))
    demark_range_multiplier: float = Field(
        default=1.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the DeMark range multiplier (e.g., 1.0): ",
            prompt_on_new=True))
    demark_level_used: str = Field(
        default="1",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the DeMark level to use for confirmation (1, 2, or 3): ",
            prompt_on_new=True))
    # SOTT parameters
    sott_period_k: int = Field(
        default=500,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the SOTT Stochastic %K Length (e.g., 500): ",
            prompt_on_new=True))
    sott_smooth_k: int = Field(
        default=200,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the SOTT Stochastic %K Smoothing (e.g., 200): ",
            prompt_on_new=True))
    sott_var_length: int = Field(
        default=2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the SOTT VAR/OTT Period (e.g., 2): ",
            prompt_on_new=True))
    sott_ott_percent: float = Field(
        default=0.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the SOTT OTT Percent (e.g., 0.5): ",
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
    Zero Lag EMA with Kalman filter and optional DeMark Pivot confirmation.
    Based on the ZLEMA indicator by M0rty.
    DeMark Pivot calculation reference: Standard technical analysis formulas.
    """

    def __init__(self, config: ZLEMAConfig, *args, **kwargs):
        self.config = config
        # Increase max_records slightly to ensure enough data for shifted calculations
        self.max_records = max(1000, config.period_slow * 3) 
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
            self.logger().warning(f"Invalid source '{self.config.source}', defaulting to 'close'.")
            return df["close"]

    def _calculate_var(self, src: pd.Series, length: int, cmo_period: int = 9) -> pd.Series:
        """
        Calculates the VAR (Volatility Adjusted Rate) indicator, mirroring the Var_Func in the SOTT Pine Script.
        This is essentially an EMA where the alpha is adjusted by the absolute value of the Chande Momentum Oscillator (CMO).
        """
        if src.isnull().all() or len(src) < cmo_period:
            return pd.Series(np.nan, index=src.index)

        valpha = 2 / (length + 1)
        src_diff = src.diff()
        
        vud = src_diff.where(src_diff > 0, 0).rolling(window=cmo_period, min_periods=1).sum()
        vdd = (-src_diff).where(src_diff < 0, 0).rolling(window=cmo_period, min_periods=1).sum()

        # Calculate CMO, handle potential division by zero
        vud_plus_vdd = vud + vdd
        vcmo = ((vud - vdd) / vud_plus_vdd.replace(0, 1e-9)).fillna(0) # Use 1e-9 to avoid true zero

        # Calculate VAR iteratively (simulating Pine Script's recursive definition)
        var_series = pd.Series(np.nan, index=src.index, dtype=float)
        if not src.empty:
            first_valid_index = src.first_valid_index()
            if first_valid_index is not None:
                var_series[first_valid_index] = src[first_valid_index] # Initialize VAR with the first source value

                alpha_cmo_abs = valpha * vcmo.abs()
                
                # Iterative calculation loop (vectorized where possible, but the core depends on the previous VAR value)
                # Using pandas iteration for clarity, performance might degrade on huge datasets
                # For larger datasets, consider numba or cython if this becomes a bottleneck
                for i in range(src.index.get_loc(first_valid_index) + 1, len(src)):
                    prev_var = var_series.iloc[i-1]
                    current_alpha = alpha_cmo_abs.iloc[i]
                    current_src = src.iloc[i]
                    
                    if pd.isna(prev_var) or pd.isna(current_alpha) or pd.isna(current_src):
                         # Attempt to forward fill from previous if current alpha/src is bad, otherwise keep NaN
                         var_series.iloc[i] = prev_var if not pd.isna(prev_var) else np.nan
                    else:
                         var_series.iloc[i] = (current_alpha * current_src) + (1 - current_alpha) * prev_var
            
        # Fill initial NaNs often caused by CMO calculation window
        return var_series.fillna(method='ffill')

    def kalman_filter(self, src: pd.Series) -> pd.Series:
        """
        Enhanced Kalman filter with adaptive measurement and process noise
        """
        if src.isnull().all() or len(src) < 2:
            return src
        
        # Calculate volatility-based measurement noise
        returns = src.pct_change().dropna()
        volatility = returns.rolling(window=20, min_periods=5).std().fillna(0.01)
        latest_volatility = volatility.iloc[-1] if not volatility.empty else 0.01
        
        # Adaptive measurement noise based on recent volatility
        measurement_noise_variance = src.diff().dropna().var()
        if pd.isna(measurement_noise_variance) or measurement_noise_variance == 0:
            measurement_noise_variance = 0.01
        
        # Adaptive process noise - more responsive during high volatility periods
        process_noise_variance = 1e-5 * (1 + (latest_volatility * 100))
        
        # Initialization
        x_hat = src.iloc[0]
        p = 1.0
        
        filtered_series = pd.Series(index=src.index, dtype=float)
        
        for i, z in enumerate(src):
            if pd.isna(z):
                filtered_series.iloc[i] = x_hat
                p = p + process_noise_variance
                continue
            
            # Prediction step
            x_hat_minus = x_hat
            p_minus = p + process_noise_variance
            
            # Update step
            kalman_gain = p_minus / (p_minus + measurement_noise_variance)
            x_hat = x_hat_minus + kalman_gain * (z - x_hat_minus)
            p = (1 - kalman_gain) * p_minus
            
            filtered_series.iloc[i] = x_hat
        
        return filtered_series

    def zlema(self, src: pd.Series, period: int) -> pd.Series:
        """
        Enhanced Zero Lag EMA with adaptive lag parameter based on recent volatility
        """
        # Calculate recent volatility
        volatility = src.pct_change().abs().rolling(window=20, min_periods=5).mean().fillna(0.01)
        
        # Adjust lag dynamically based on volatility - less lag during steady trends
        avg_volatility = volatility.mean()
        volatility_ratio = volatility / avg_volatility
        
        # Calculate adaptive lag (bounded between 20% and 100% of original)
        adaptive_ratio = np.clip(1.0 - (volatility_ratio - 1.0) * 0.2, 0.2, 1.0)
        base_lag = int((period - 1) / 2)
        adaptive_lag = np.maximum(1, np.floor(base_lag * adaptive_ratio.iloc[-1]))
        
        # Apply standard ZLEMA with adaptive lag
        ema_data = src + (src - src.shift(int(adaptive_lag)).fillna(src))
        return ema_data.ewm(span=period, adjust=False).mean()

    def calculate_demark_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced DeMark Pivot Points calculation with additional levels
        """
        # Use the configured lookback period
        lookback = self.config.demark_lookback_period
        range_multiplier = self.config.demark_range_multiplier
        
        prev_high = df['high'].shift(lookback)
        prev_low = df['low'].shift(lookback)
        prev_close = df['close'].shift(lookback)
        prev_open = df['open'].shift(lookback)

        conditions = [
            prev_close < prev_open,
            prev_close > prev_open,
            prev_close == prev_open
        ]
        choices_x = [
            prev_high + 2 * prev_low + prev_close,
            2 * prev_high + prev_low + prev_close,
            prev_high + prev_low + 2 * prev_close
        ]

        x = np.select(conditions, choices_x, default=np.nan)

        df['demark_p'] = x / 4
        df['demark_r1'] = x / 2 - prev_low
        df['demark_s1'] = x / 2 - prev_high
        
        # Add R2, R3, S2, S3 for more precise support/resistance identification
        # Apply the range multiplier to adjust sensitivity
        range_value = (prev_high - prev_low) * range_multiplier
        df['demark_r2'] = df['demark_p'] + range_value
        df['demark_r3'] = df['demark_r1'] + range_value
        df['demark_s2'] = df['demark_p'] - range_value
        df['demark_s3'] = df['demark_s1'] - range_value
        return df

    def _calculate_sott(self, df: pd.DataFrame):
        """
        Calculates the Stochastic Optimized Trend Tracker (SOTT) indicators.
        Based on the Pine Script by KivancOzbilgic/Anil_Ozeksi.
        Adds 'stoch_k', 'k', 'MAvg', 'longStop', 'shortStop', 'dir', 'MT', 'OTT' columns to the DataFrame.
        """
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            self.logger().warning("SOTT calculation requires 'high', 'low', 'close' columns. Skipping.")
            return df   

        # Calculate Stochastic %K
        stoch = df.ta.stoch(k=self.config.sott_period_k, d=3, smooth_k=3) # Standard calculation, d/smooth_k don't matter for %K
        if stoch is None or f'STOCHk_{self.config.sott_period_k}_3_3' not in stoch.columns:
             self.logger().warning("Failed to calculate Stochastic %K. Skipping SOTT.")
             return df
        df['stoch_k'] = stoch[f'STOCHk_{self.config.sott_period_k}_3_3']

        # Calculate 'k' (Smoothed, VAR-adjusted Stochastic)
        # Using sott_smooth_k as the 'length' for VAR function applied to stoch_k
        df['k'] = self._calculate_var(df['stoch_k'], length=self.config.sott_smooth_k)

        # Check if 'k' calculation was successful
        if df['k'].isnull().all():
            self.logger().warning("SOTT 'k' calculation resulted in all NaNs. Skipping further SOTT steps.")
            return df
        
        # Calculate 'MAvg' (VAR applied to 'k')
        # Using sott_var_length as the 'length' for VAR function applied to k
        df['MAvg'] = self._calculate_var(df['k'], length=self.config.sott_var_length)

        # Check if 'MAvg' calculation was successful
        if df['MAvg'].isnull().all():
            self.logger().warning("SOTT 'MAvg' calculation resulted in all NaNs. Skipping further SOTT steps.")
            return df

        # Calculate OTT components
        fark = df['MAvg'] * self.config.sott_ott_percent * 0.01
        df['longStop'] = df['MAvg'] - fark
        df['shortStop'] = df['MAvg'] + fark

        # Apply trailing stop logic iteratively (simulating Pine Script's := and nz behavior)
        longStopPrev = df['longStop'].shift(1).ffill()
        df['longStop'] = np.where(df['MAvg'] > longStopPrev, np.maximum(df['longStop'], longStopPrev), df['longStop'])

        shortStopPrev = df['shortStop'].shift(1).ffill()
        df['shortStop'] = np.where(df['MAvg'] < shortStopPrev, np.minimum(df['shortStop'], shortStopPrev), df['shortStop'])
        
        # Fill initial NaNs that might arise from the shift/comparison
        df['longStop'] = df['longStop'].ffill()
        df['shortStop'] = df['shortStop'].ffill()

        # Calculate direction ('dir') iteratively
        df['dir'] = 1 # Start with default direction 1
        dir_prev = df['dir'].shift(1).fillna(1) # Use 1 if previous is NaN
        mavg_lt_longstop_prev = df['MAvg'] < df['longStop'].shift(1).ffill()
        mavg_gt_shortstop_prev = df['MAvg'] > df['shortStop'].shift(1).ffill()

        # Define conditions for dir change
        # dir := dir == -1 and MAvg > shortStopPrev ? 1 : dir == 1 and MAvg < longStopPrev ? -1 : dir
        # Need to calculate iteratively or use a mask-based approach that propagates state
        
        # Iterative approach for 'dir' (simpler to implement correctly)
        dir_series = pd.Series(1, index=df.index, dtype=int) # Initialize with 1
        first_valid_idx = df[['MAvg', 'longStop', 'shortStop']].dropna().index.min()

        if first_valid_idx is not None:
            start_loc = df.index.get_loc(first_valid_idx)
            if start_loc > 0: # Need at least one previous row for comparison
                 # Fill initial values before loop correctly
                 dir_series.iloc[:start_loc] = 1 # Or potentially NaN if strict
                 
                 current_dir = 1 # Initialize based on first valid location logic if needed, or default 1
                 dir_series.iloc[start_loc] = current_dir # Set the first calculable dir

                 for i in range(start_loc + 1, len(df)):
                     prev_dir = dir_series.iloc[i-1]
                     mavg_val = df['MAvg'].iloc[i]
                     long_stop_prev_val = df['longStop'].iloc[i-1] # Use actual previous value
                     short_stop_prev_val = df['shortStop'].iloc[i-1] # Use actual previous value

                     if pd.isna(mavg_val) or pd.isna(long_stop_prev_val) or pd.isna(short_stop_prev_val):
                          dir_series.iloc[i] = prev_dir # Maintain previous dir if data is missing
                          continue

                     if prev_dir == -1 and mavg_val > short_stop_prev_val:
                         current_dir = 1
                     elif prev_dir == 1 and mavg_val < long_stop_prev_val:
                         current_dir = -1
                     else:
                         current_dir = prev_dir # No change
                     dir_series.iloc[i] = current_dir
            df['dir'] = dir_series
        else:
             df['dir'] = 1 # Fallback if no valid data


        # Calculate MT and OTT
        df['MT'] = np.where(df['dir'] == 1, df['longStop'], df['shortStop'])
        # Pine script plots OTT[2] for signals - we store OTT but will use OTT.shift(2) in signal logic
        # OTT=MAvg>MT ? MT*(200+percent)/200 : MT*(200-percent)/200
        # This calculation in pine seems off, it compares MAvg with MT but then modifies MT based on that comparison?
        # Let's replicate directly first:
        ott_factor = np.where(df['MAvg'] > df['MT'], (200 + self.config.sott_ott_percent) / 200, (200 - self.config.sott_ott_percent) / 200)
        df['OTT'] = df['MT'] * ott_factor
        
        # Alternative interpretation: OTT = MAvg if MAvg crosses MT based on dir? Seems unlikely given plot(OTT[2])
        # Let's stick to direct replication of the formula found.

        return df

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        if df.empty:
             self.logger().warning("Candles data is empty. Skipping processing.")
             self.processed_data["signal"] = 0
             self.processed_data["features"] = df
             return

        # Get price source
        src = self.get_price_source(df)

        # Apply Kalman filter (always enabled)
        src_filtered = self.kalman_filter(src.copy())  # Use .copy() to avoid modifying original df source indirectly
        # Only use filtered source if the filter ran successfully (returned non-empty series)
        if not src_filtered.isnull().all():
            src = src_filtered
        else:
            self.logger().warning("Kalman filter returned empty or all NaN series. Using original source.")

        # Calculate ZLEMAs
        ma1 = self.zlema(src, self.config.period_fast)
        ma2 = self.zlema(src, self.config.period_medium)
        ma3 = self.zlema(src, self.config.period_slow)

        # Store calculated values
        df['zlema_fast'] = ma1
        df['zlema_medium'] = ma2
        df['zlema_slow'] = ma3
        df['kalman_filtered_src'] = src  # Store the source used for ZLEMA calculation (filtered or original)

        # --- Base Signal Calculation ---
        base_signal = pd.Series(0, index=df.index)  # Initialize base signal
        
        # ZLEMA Crossover Logic (always used)
        if len(ma1) > 1 and len(ma2) > 1:
            buy_signal_active = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))
            sell_signal_active = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))
            base_signal[buy_signal_active.fillna(False)] = 1
            base_signal[sell_signal_active.fillna(False)] = -1
        else:
            self.logger().warning("Not enough data points to calculate ZLEMA crosses for base signal.")

        df['base_signal'] = base_signal # Store base signal for reference

        # --- Final Signal Generation with DeMark and SOTT Confirmation ---
        final_signal = pd.Series(0, index=df.index)  # Initialize final signal

        # --- Confirmation Logic ---
        # Both confirmations are always enabled
        sott_buy = sott_sell = demark_buy = demark_sell = None

        # Calculate SOTT confirmation
        df = self._calculate_sott(df)
        if 'k' in df and 'OTT' in df and not df['k'].isnull().all() and not df['OTT'].isnull().all():
            ott_shifted = df['OTT'].shift(2)
            if not ott_shifted.isnull().all():
                sott_buy = (base_signal == 1) & (df['k'] > ott_shifted) & (df['k'].shift(1) <= ott_shifted.shift(1))
                sott_sell = (base_signal == -1) & (df['k'] < ott_shifted) & (df['k'].shift(1) >= ott_shifted.shift(1))
            else:
                self.logger().warning("Could not apply SOTT confirmation trigger due to insufficient shifted OTT data.")
                sott_buy = sott_sell = None
        else:
            self.logger().warning("Could not apply SOTT confirmation trigger due to missing 'k' or 'OTT' data or all NaNs.")
            sott_buy = sott_sell = None

        # Calculate DeMark confirmation
        df = self.calculate_demark_pivots(df)
        
        # Use the configured DeMark level for confirmation
        demark_level = self.config.demark_level_used
        resistance_col = f'demark_r{demark_level}'  # e.g. 'demark_r1', 'demark_r2', 'demark_r3'
        support_col = f'demark_s{demark_level}'     # e.g. 'demark_s1', 'demark_s2', 'demark_s3'
        
        if 'high' in df and 'low' in df and resistance_col in df and support_col in df:
            demark_buy = (base_signal == 1) & (df['high'] > df[resistance_col])
            demark_sell = (base_signal == -1) & (df['low'] < df[support_col])
        else:
            self.logger().warning(f"Could not apply DeMark Pivot breakout trigger due to missing data (high, low, {resistance_col}, or {support_col}).")
            demark_buy = demark_sell = None

        # AND logic: require both confirmations
        if sott_buy is not None and demark_buy is not None:
            buy_trigger = sott_buy & demark_buy
            final_signal[buy_trigger.fillna(False)] = 1
        if sott_sell is not None and demark_sell is not None:
            sell_trigger = sott_sell & demark_sell
            final_signal[sell_trigger.fillna(False)] = -1

        df['signal'] = final_signal  # Store final signal

        # Update processed data with the latest final signal and features
        self.processed_data["signal"] = df["signal"].iloc[-1] if not df["signal"].empty and not pd.isna(df["signal"].iloc[-1]) else 0
        self.processed_data["features"] = df