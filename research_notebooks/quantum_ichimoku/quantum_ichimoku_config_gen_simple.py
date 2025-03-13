from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop

from controllers.directional_trading.quantum_ichimoku import QuantumIchimokuControllerConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class QuantumIchimokuConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("1000")

    async def generate_config(self, trial) -> BacktestingConfig:
        # Basic parameters
        interval = "1m"          
        
        # MACD parameters
        macd_fast = trial.suggest_int("macd_fast", 10, 30, step=2)
        macd_slow = trial.suggest_int("macd_slow", macd_fast + 5, 55, step=5)
        macd_signal = trial.suggest_int("macd_signal", 6, 22, step=2)
        
        # EMA parameters
        ema_short = trial.suggest_int("ema_short", 4, 16, step=2)
        ema_medium = trial.suggest_int("ema_medium", ema_short + 2, ema_short + 20, step=2)
        ema_long = trial.suggest_int("ema_long", ema_medium + 2, ema_medium + 20, step=2)
        
        # ATR parameters
        atr_length = trial.suggest_int("atr_length", 5, 20, step=1)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.0, 3.0, step=0.1)
        
        # Ichimoku Cloud parameters
        tenkan_period = trial.suggest_int("tenkan_period", 7, 11, step=1)
        kijun_period = trial.suggest_int("kijun_period", 22, 30, step=2)
        senkou_span_b_period = trial.suggest_int("senkou_span_b_period", 44, 60, step=4)
        displacement = trial.suggest_int("displacement", 22, 30, step=2)
        
        # Volume parameters
        volume_ma_period = trial.suggest_int("volume_ma_period", 15, 25, step=1)
        
        # New Quantum-specific parameters
        rsi_period = trial.suggest_int("rsi_period", 10, 20, step=2)
        rsi_overbought = trial.suggest_int("rsi_overbought", 65, 80, step=5)
        rsi_oversold = trial.suggest_int("rsi_oversold", 20, 35, step=5)
        
        bb_length = trial.suggest_int("bb_length", 15, 25, step=2)
        bb_std = trial.suggest_float("bb_std", 1.8, 2.5, step=0.1)
        
        vwap_length = trial.suggest_int("vwap_length", 10, 20, step=2)
        
        adx_length = trial.suggest_int("adx_length", 10, 20, step=2)
        adx_threshold = trial.suggest_int("adx_threshold", 20, 30, step=2)

        # Triple barrier metrics
        take_profit = trial.suggest_float("take_profit", 0.5, 0.6, step=0.1)
        stop_loss = trial.suggest_float("stop_loss", 0.005, 0.01, step=0.001)
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.002, 0.01, step=0.001)
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.001, 0.02, step=0.0005)
        max_executors_per_side = trial.suggest_int("max_executors_per_side", 1, 5)

        # Id Generation
        controller_id = (f"quantum_ichimoku_{self.connector_name}_{interval}_{self.trading_pair}_"
                       f"macd_{macd_fast}_{macd_slow}_{macd_signal}_"
                       f"ema_{ema_short}_{ema_medium}_{ema_long}_"
                       f"atr_{atr_length}_{atr_multiplier}_"
                       f"ichi_{tenkan_period}_{kijun_period}_{senkou_span_b_period}_"
                       f"rsi_{rsi_period}_{rsi_overbought}_{rsi_oversold}_"
                       f"bb_{bb_length}_{bb_std}_"
                       f"adx_{adx_length}_{adx_threshold}")

        # Create the strategy configuration
        config = QuantumIchimokuControllerConfig(
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
            atr_multiplier=Decimal(str(atr_multiplier)),
            tenkan_period=tenkan_period,
            kijun_period=kijun_period,
            senkou_span_b_period=senkou_span_b_period,
            displacement=displacement,
            volume_ma_period=volume_ma_period,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            bb_length=bb_length,
            bb_std=Decimal(str(bb_std)),
            vwap_length=vwap_length,
            adx_length=adx_length,
            adx_threshold=adx_threshold,
            take_profit=Decimal(str(take_profit)),
            stop_loss=Decimal(str(stop_loss)),
            trailing_stop=TrailingStop(
                activation_price=Decimal(str(trailing_stop_activation_price)),
                trailing_delta=Decimal(str(trailing_stop_trailing_delta))
            ),
            max_executors_per_side=max_executors_per_side,
            time_limit=60 * 60 * 2,
            cooldown_time=60,
        )
        
        return BacktestingConfig(config=config, start=self.start, end=self.end)
