from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.elitesmugplug import SmugPlug3LiteConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class SmugPlug3LiteConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate specific SmugPlug3Lite metrics
        interval = "1m"  # Fixed 1m interval
        # EMA parameters with new ranges matching 3lite version
        ema_fast = trial.suggest_int("ema_fast", 6, 10)  # Centered around 8
        ema_medium = trial.suggest_int("ema_medium", 18, 24)  # Centered around 21
        ema_long = trial.suggest_int("ema_long", 31, 37)  # Centered around 34
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 8, 12)  # Centered around 10
        atr_multiplier = trial.suggest_float("atr_multiplier", 0.7, 1.1, step=0.1)  # Centered around 0.9
        
        # New 3lite specific parameters
        volume_surge = trial.suggest_float("volume_surge", 2.8, 3.4, step=0.1)  # Centered around 3.1
        rsi_period = trial.suggest_int("rsi_period", 11, 15)  # Centered around 13
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.26, 0.32, step=0.01)  # Centered around 0.29
        stop_loss = trial.suggest_float("stop_loss", 0.04, 0.08, step=0.005)  # Centered around 0.06
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.02, 0.03, step=0.001)  # Centered around 0.025
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.016, 0.022, step=0.001)  # Centered around 0.019
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)  # Centered around 2

        # Id Generation
        controller_id = (f"3lite_smugplug_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"ema_{ema_fast}_{ema_medium}_{ema_long}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"vol_{volume_surge}_rsi_{rsi_period}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = SmugPlug3LiteConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval=interval,
            ema_fast=ema_fast,
            ema_medium=ema_medium,
            ema_slow=ema_long,
            atr_length=atr_length,
            atr_multiplier=Decimal(atr_multiplier),
            volume_surge=Decimal(volume_surge),
            rsi_period=rsi_period,
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
