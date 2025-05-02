from decimal import Decimal
import datetime
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop
from optuna import trial

from controllers.directional_trading.zlem import ZLEMAConfig
from core.backtesting.optimizer import BacktestingConfig, BaseStrategyConfigGenerator


class ZLEMAConfigGenerator(BaseStrategyConfigGenerator):
    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        super().__init__(start_date, end_date)
        self.connector_name = "binance_perpetual"
        self.trading_pair = "WLD-USDT"
        self.total_amount_quote = Decimal("300")  # Reduced amount for more conservative testing

    async def generate_config(self, trial) -> BacktestingConfig:
        # Fixed parameters based on best practices
        max_executors_per_side = 1  # Limit risk exposure
        time_limit = 7200  # 2 hours time limit
        cooldown_time = 60  # 1 minute cooldown between trades
        
        source = trial.suggest_categorical("source", ["open", "high", "low", "close", "hlc3", "ohlc4"])  # Try different price sources

        # Tunable parameters with narrower ranges based on successful configurations
        period_fast = trial.suggest_int("period_fast", 10, 20)  # Centered around 15
        # Ensure medium > fast
        period_medium = trial.suggest_int("period_medium", max(period_fast + 1, 11), 25)
        # Ensure slow > medium
        period_slow = trial.suggest_int("period_slow", max(period_medium + 1, 75), 95)  # Centered around 85

        # --- Confirmation Logic ---
        # Both SOTT and DeMark confirmations are always enabled
        
        # Define SOTT parameters
        sott_period_k = trial.suggest_int("sott_period_k", 400, 600)
        sott_smooth_k = trial.suggest_int("sott_smooth_k", 150, 250)
        sott_var_length = trial.suggest_int("sott_var_length", 2, 5)
        sott_ott_percent = trial.suggest_float("sott_ott_percent", 0.3, 0.7, step=0.1)
        
        # Define DeMark parameters
        demark_lookback_period = trial.suggest_int("demark_lookback_period", 1, 3)
        demark_range_multiplier = trial.suggest_float("demark_range_multiplier", 0.8, 1.5, step=0.1)
        demark_level_used = trial.suggest_categorical("demark_level_used", ["1", "2", "3"])
        
        # Create combined confirmation type for ID
        confirmation_type = f"SOTT_DeMark_S{sott_period_k}_{sott_smooth_k}_{sott_var_length}_{sott_ott_percent}_D{demark_lookback_period}_{demark_range_multiplier}_{demark_level_used}"

        # Triple barrier metrics with optimized ranges
        take_profit = 0.3  # Fixed take profit at 0.3
        stop_loss = 0.04  # Fixed stop loss at 0.04
        trailing_stop_activation_price = trial.suggest_float("trailing_stop_activation_price", 0.005, 0.025, step=0.001)  # Centered around 0.02
        trailing_stop_trailing_delta = trial.suggest_float("trailing_stop_trailing_delta", 0.0015, 0.0025, step=0.0001)  # Centered around 0.002

        # Id Generation - Include confirmation type
        controller_id = (f"zlem_{self.connector_name}_{self.trading_pair}_"
                       f"periods_{period_fast}_{period_medium}_{period_slow}_"
                       f"confirm_{confirmation_type}_" 
                       f"tp{round(100 * take_profit, 1)}_"
                       f"sl{round(100 * stop_loss, 1)}_"
                       f"ts{round(100 * trailing_stop_activation_price, 1)}-"
                       f"{round(1000 * trailing_stop_trailing_delta, 1)}")

        # Create the strategy configuration with updated parameters
        config = ZLEMAConfig(
            id=controller_id,
            total_amount_quote=self.total_amount_quote,
            connector_name=self.connector_name,
            trading_pair=self.trading_pair,
            interval="1m",
            source=source,
            period_fast=period_fast,
            period_medium=period_medium,
            period_slow=period_slow,
            # Pass SOTT parameters
            sott_period_k=sott_period_k,
            sott_smooth_k=sott_smooth_k,
            sott_var_length=sott_var_length,
            sott_ott_percent=sott_ott_percent,
            # Pass DeMark parameters
            demark_lookback_period=demark_lookback_period,
            demark_range_multiplier=demark_range_multiplier,
            demark_level_used=demark_level_used,
            take_profit=Decimal(take_profit),
            stop_loss=Decimal(stop_loss),
            trailing_stop=TrailingStop(
                activation_price=Decimal(trailing_stop_activation_price),
                trailing_delta=Decimal(trailing_stop_trailing_delta)
            ),
            max_executors_per_side=max_executors_per_side,
            time_limit=time_limit,
            cooldown_time=cooldown_time,
        )
        # Return the configuration encapsulated in BacktestingConfig
        return BacktestingConfig(config=config, start=self.start, end=self.end)
