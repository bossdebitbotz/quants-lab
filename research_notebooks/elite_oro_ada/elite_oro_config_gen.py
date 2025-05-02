from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.elite_oro_1 import EliteOroConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class EliteOroConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "ADA-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate specific EliteOro metrics
        interval = "1m"  # Default interval for EliteOro
        
        # ZLEMA parameters based on EliteOro ranges
        zlema_fast = trial.suggest_int("zlema_fast", 3, 8)  # Default is 4
        zlema_medium = trial.suggest_int("zlema_medium", 8, 21)  # Default is 12
        zlema_slow = trial.suggest_int("zlema_slow", 13, 34)  # Default is 13
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 10, 21)  # Default is 20
        atr_multiplier = trial.suggest_float("atr_multiplier", 0.8, 1.5, step=0.1)  # Default is 1.2
        
        # Volume parameters
        volume_ma_period = trial.suggest_int("volume_ma_period", 15, 25)  # Default is 20
        volume_surge = trial.suggest_float("volume_surge", 1.5, 3.0, step=0.1)  # Default is 1.6
        
        # Price channel and RSI parameters
        price_channel_period = trial.suggest_int("price_channel_period", 5, 20)  # Default is 10
        rsi_period = trial.suggest_int("rsi_period", 8, 21)  # Default is 14
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.26, 0.32, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.04, 0.08, step=0.005)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.02, 0.03, step=0.001)
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.016, 0.022, step=0.001)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 3)

        # Id Generation
        controller_id = (f"elite_oro_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"zlema_{zlema_fast}_{zlema_medium}_{zlema_slow}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"vol_{volume_ma_period}_{volume_surge}_"
                       f"pc_{price_channel_period}_rsi_{rsi_period}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = EliteOroConfig(
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