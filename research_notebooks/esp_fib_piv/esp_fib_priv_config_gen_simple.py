from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.esp_fib_piv import ESPFibPivConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class ESPFibPivConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate specific ESPFibPiv metrics
        interval = "1m"  # Fixed 3m interval
        # EMA parameters
        # EMA parameters - using values from context as midpoints with wider spreads
        ema_fast = trial.suggest_int("ema_fast", 1, 7)  # midpoint 4
        ema_medium = trial.suggest_int("ema_medium", 12, 26)  # midpoint 19
        ema_slow = trial.suggest_int("ema_slow", 6, 20)  # midpoint 13
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 6, 18)  # midpoint 12
        atr_multiplier = trial.suggest_float("atr_multiplier", 0.8, 2.0, step=0.1)  # midpoint 1.4
        
        # ESPFibPiv specific parameters
        volume_surge = trial.suggest_float("volume_surge", 0.8, 2.0, step=0.1)  # midpoint 1.4
        rsi_period = trial.suggest_int("rsi_period", 12, 26)  # midpoint 19
        fib_pivot_period = trial.suggest_int("fib_pivot_period", 6, 20)  # midpoint 13
        fib_confirmation_threshold = trial.suggest_float("fib_confirmation_threshold", 0.4, 1.0, step=0.1)  # midpoint 0.7
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.37, 0.38, step=0.01)  # midpoint 0.28
        stop_loss = trial.suggest_float("stop_loss", 0.03, 0.12, step=0.005)  # midpoint 0.075
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.02, step=0.001)  # midpoint 0.025
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.001, 0.005, step=0.001)  # midpoint 0.018
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 5)

        # Id Generation
        controller_id = (f"esp_fib_piv_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"ema_{ema_fast}_{ema_medium}_{ema_slow}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"vol_{volume_surge}_rsi_{rsi_period}_"
                       f"fib_{fib_pivot_period}_{fib_confirmation_threshold}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = ESPFibPivConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval=interval,
            ema_fast=ema_fast,
            ema_medium=ema_medium,
            ema_slow=ema_slow,
            atr_length=atr_length,
            atr_multiplier=Decimal(atr_multiplier),
            volume_surge=Decimal(volume_surge),
            rsi_period=rsi_period,
            fib_pivot_period=fib_pivot_period,
            fib_confirmation_threshold=Decimal(fib_confirmation_threshold),
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price),
                trailing_delta=Decimal(trailing_stop_trailing_delta)
            ),
            max_executors_per_side=max_executors_per_side,
            time_limit=60 * 60 * 2,
            cooldown_time=60,
        )
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)
