from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.zlemog import ZLEMAConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class ZLEMAConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("300")  # Updated total amount

    async def generate_config(self, trial) -> BacktestingConfig:
        # Fixed parameters based on central configuration
        source = "close"
        show_cross = True
        max_executors_per_side = 1
        time_limit = 7200
        cooldown_time = 60
        
        # Add higher timeframe interval for validation
        higher_interval = trial.suggest_categorical("higher_interval", ["5m", "15m", "30m", "1h", "4h"])
        # Tunable parameters centered around the provided values
        period_fast = trial.suggest_int("period_fast", 10, 20) # Centered around 15
        # Ensure medium > fast, centered around 16
        period_medium = trial.suggest_int("period_medium", max(period_fast + 1, 11), 25)
        # Ensure slow > medium, centered around 85
        period_slow = trial.suggest_int("period_slow", max(period_medium + 1, 75), 95)

        # Triple barrier metrics centered around provided values
        take_profit = trial.suggest_float("take_profit", 0.35, 0.39, step=0.005) # Centered around 0.37
        stop_loss = trial.suggest_float("stop_loss", 0.04, 0.05, step=0.001) # Centered around 0.045
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.015, 0.025, step=0.001) # Centered around 0.02
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.0015, 0.0025, step=0.0001) # Centered around 0.002

        # Id Generation - Using base "zlem_1" and only variable params
        controller_id = (f"zlemog_{self.connector_name}_{self.trading_pair}_"
                       f"periods_{period_fast}_{period_medium}_{period_slow}_"
                       f"tp{round(100 * take_profit, 1)}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(1000 * trailing_stop_trailing_delta, 1)}") # Adjusted precision for delta in ID

        # Create the strategy configuration with fixed and tuned parameters
        config = ZLEMAConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval="1m",
            higher_interval=higher_interval,  # Added higher timeframe interval
            source=source,
            period_fast=period_fast, # Tuned
            period_medium=period_medium, # Tuned
            period_slow=period_slow, # Tuned 
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
