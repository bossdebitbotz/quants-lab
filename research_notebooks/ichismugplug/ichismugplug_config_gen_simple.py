from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.ichismugplug import IchimokuSmugPlugControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class IchimokuSmugPlugConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Generate specific IchimokuSmugPlug metrics
        interval = "1m"          
        # MACD parameters
        macd_fast = trial.suggest_int("macd_fast", 10, 30, step=5)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 5, 55, step=5)
        macd_signal = trial.suggest_int("macd_signal", 6, 22, step=2)
        
        # EMA parameters (ensure short < medium < long)
        ema_short = trial.suggest_int("ema_short", 4, 16, step=2)
        ema_medium = trial.suggest_int("ema_medium", ema_short + 2, ema_short + 20, step=2)
        ema_long = trial.suggest_int("ema_long", ema_medium + 2, ema_medium + 20, step=2)
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 5, 20, step=1)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0, step=0.1)
        
        # Ichimoku Cloud parameters
        tenkan_period = trial.suggest_int("tenkan_period", 7, 11, step=1)  # centered around 9
        kijun_period = trial.suggest_int("kijun_period", 22, 30, step=2)   # centered around 26
        senkou_span_b_period = trial.suggest_int("senkou_span_b_period", 44, 60, step=4)  # centered around 52
        displacement = trial.suggest_int("displacement", 22, 30, step=2)    # centered around 26
        
        # Volume parameters
        volume_ma_period = trial.suggest_int("volume_ma_period", 15, 25, step=1)  # centered around 20
        
        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.01, 0.5, step=0.01)
        stop_loss = trial.suggest_float("stop_loss", 0.005, 0.1, step=0.005)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.05, step=0.001)
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.001, 0.02, step=0.0005)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 5)

        # Id Generation
        controller_id = (f"ichismugplug_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"macd_{macd_fast}_{macd_slow}_{macd_signal}_"
                       f"ema_{ema_short}_{ema_medium}_{ema_long}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"ichi_{tenkan_period}_{kijun_period}_{senkou_span_b_period}_"
                       f"vol_{volume_ma_period}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(100 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration
        config = IchimokuSmugPlugControllerConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval=interval,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            ema_short=ema_short,
            ema_medium=ema_medium,
            ema_long=ema_long,
            atr_length=atr_length,
            atr_multiplier=Decimal(atr_multiplier),
            tenkan_period=tenkan_period,
            kijun_period=kijun_period,
            senkou_span_b_period=senkou_span_b_period,
            displacement=displacement,
            volume_ma_period=volume_ma_period,
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
