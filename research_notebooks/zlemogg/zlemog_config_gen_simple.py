from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial
import os

from controllers.directional_trading.zlemogg import ZLEMAConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class ZLEMAConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = "1000"  # Updated total amount
        # Path to the candles data file


    async def generate_config(self, trial) -> BacktestingConfig:
        # Fixed parameters based on central configuration
    
        max_executors_per_side = 1
        # Time limit options in seconds for optimization trials
        time_limit = trial.suggest_categorical("time_limit", [1800, 3600, 7200, 14400])  # 30min, 1hr, 2hr, 4hr
        cooldown_time = 60
        
        # Source options: "open", "high", "low", "close", "hlc3", "ohlc4"
        source = trial.suggest_categorical("source", ["open", "high", "low", "close", "hlc3", "ohlc4"])

        # Tunable parameters centered around the provided values
        period_fast = trial.suggest_int("period_fast", 5, 25) # Wider range around 15
        # Ensure medium > fast, with wider range
        period_medium = trial.suggest_int("period_medium", max(period_fast + 1, 10), 35)
        # Ensure slow > medium, with wider range
        period_slow = trial.suggest_int("period_slow", max(period_medium + 1, 50), 120)

        # Triple barrier metrics centered around provided values
        take_profit = 0.3  # Fixed take profit value
        stop_loss = 0.05  # Fixed stop loss value
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.015, step=0.01) # Centered around 0.02
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.003, 0.005, step=0.001) # Centered around 0.002

        # Fractal parameters
        fractal_window = trial.suggest_int("fractal_window", 3, 10)  # Range from minimum viable (3) to max useful (10)
        fractal_lookback_period = trial.suggest_int("fractal_lookback_period", 5, 30)  # How far back to check for confirmation

        # CVIX parameters
        cvix_period = trial.suggest_int("cvix_period", 20, 50)  # Period for CVIX calculation
        cvix_threshold_low = trial.suggest_float("cvix_threshold_low", 0.2, 0.4, step=0.05)  # Low volatility threshold
        cvix_threshold_high = trial.suggest_float("cvix_threshold_high", 0.6, 0.8, step=0.05)  # High volatility threshold

        # Id Generation - Using base "zlem_1" and only variable params
        controller_id = (f"zlemogg_{self.connector_name}_{self.trading_pair}_"
                       f"periods_{period_fast}_{period_medium}_{period_slow}_"
                       f"tp{round(100 * take_profit, 1)}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(1000 * trailing_stop_trailing_delta, 1)}"
                       f"_fw{fractal_window}_flb{fractal_lookback_period}"
                       f"_cvix{cvix_period}_{int(cvix_threshold_low*100)}_{int(cvix_threshold_high*100)}") # Add CVIX parameters to ID

        # Create the strategy configuration with fixed and tuned parameters
        config = ZLEMAConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval="1m",
            source=source, # Fixed
            period_fast=period_fast, # Tuned
            period_medium=period_medium, # Tuned
            period_slow=period_slow, # Tuned
            fractal_window=fractal_window, # Tuned
            fractal_lookback_period=fractal_lookback_period, # Tuned
            cvix_period=cvix_period, # Tuned
            cvix_threshold_low=cvix_threshold_low, # Tuned
            cvix_threshold_high=cvix_threshold_high, # Tuned
            take_profit=Decimal(take_profit), # Tuned
            stop_loss=Decimal(stop_loss), # Tuned
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price), # Tuned
                trailing_delta=Decimal(trailing_stop_trailing_delta) # Tuned
            ),
            max_executors_per_side=max_executors_per_side, # Fixed
            time_limit=time_limit, # Fixed
            cooldown_time=cooldown_time, # Fixed
        )
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)
