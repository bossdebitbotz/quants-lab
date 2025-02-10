from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.eliteelite import EliteEliteConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class EliteEliteConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate EliteElite specific parameters
        interval = "1h"  # Default 1h interval
        
        # EMA parameters
        ema_short = trial.suggest_int("ema_short", 8, 13)
        ema_medium = trial.suggest_int("ema_medium", 21, 34)
        ema_long = trial.suggest_int("ema_long", 34, 55)
        
        # MACD parameters
        macd_fast = trial.suggest_int("macd_fast", 12, 26)
        macd_slow = trial.suggest_int("macd_slow", 26, 40)
        macd_signal = trial.suggest_int("macd_signal", 9, 17)
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 3, 21)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 2.0, step=0.1)
        
        # Volume and orderbook parameters
        orderbook_imbalance_threshold = trial.suggest_float("orderbook_imbalance", 0.1, 0.5, step=0.05)
        volume_surge_threshold = trial.suggest_float("volume_surge", 1.2, 3.0, step=0.1)
        volume_ma_length = trial.suggest_int("volume_ma_length", 10, 30)
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.01, 0.5, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.005, 0.1, step=0.005)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.05, step=0.001)
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.001, 0.02, step=0.0005)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 5)

        # Id Generation
        controller_id = (f"elite_elite_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"ema_{ema_short}_{ema_medium}_{ema_long}_"
                       f"macd_{macd_fast}_{macd_slow}_{macd_signal}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"vol_{volume_surge_threshold}_ob_{orderbook_imbalance_threshold}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = EliteEliteConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval=interval,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            atr_length=atr_length,
            atr_multiplier=Decimal(atr_multiplier),
            orderbook_imbalance_threshold=Decimal(orderbook_imbalance_threshold),
            volume_surge_threshold=Decimal(volume_surge_threshold),
            volume_ma_length=volume_ma_length,
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price),
                trailing_delta=Decimal(trailing_stop_trailing_delta)
            ),
            max_executors_per_side=max_executors_per_side,
            time_limit=60 * 60 * 2,  # 2 hours
            cooldown_time=60,
        )
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)
