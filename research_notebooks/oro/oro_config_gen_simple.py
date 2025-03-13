from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.oro import OroControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class OroConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate specific Oro metrics
        interval = "1m"  # Fixed interval for optimal trading performance
        
        # ZLEMA parameters - ensuring zlema_fast < zlema_medium < zlema_slow
        zlema_fast = trial.suggest_int("zlema_fast", 3, 8)
        zlema_medium = trial.suggest_int("zlema_medium", 8, 21)
        zlema_slow = trial.suggest_int("zlema_slow", 13, 34)
        
        atr_length = trial.suggest_int("atr_length", 10, 21)
        atr_multiplier = trial.suggest_float("atr_multiplier", 0.8, 1.5, step=0.1)
        
        # New parameters for updated Oro strategy
        volume_ma_period = trial.suggest_int("volume_ma_period", 10, 30)
        volume_surge = trial.suggest_float("volume_surge", 1.5, 3.0, step=0.1)
        price_channel_period = trial.suggest_int("price_channel_period", 5, 20)
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.01, 0.05, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.005, 0.05, step=0.005)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.05, step=0.001)
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.001, 0.02, step=0.0005)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 5)

        # Id Generation
        controller_id = (f"oro_{self.connector_name}_{interval}_{self.trading_pair}_"
                         f"zlema_{zlema_fast}_{zlema_medium}_{zlema_slow}_"
                         f"atr_{atr_length}_{atr_multiplier}_"
                         f"vsurge_{volume_surge}_pchan_{price_channel_period}_"
                         f"sl{round(100 * stop_loss, 1)}_"
                         f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                         f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = OroControllerConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval=interval,
            zlema_fast=zlema_fast,
            zlema_medium=zlema_medium,
            zlema_slow=zlema_slow,
            atr_length=atr_length,
            atr_multiplier=Decimal(atr_multiplier),
            volume_ma_period=volume_ma_period,
            volume_surge=Decimal(volume_surge),
            price_channel_period=price_channel_period,
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
