from typing import List, Tuple

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator
import pandas as pd
import uuid

# These functions are not currently being used in the Ichimoku strategy implementation.
# The strategy relies on Ichimoku cloud indicators (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span)
# and volume confirmation, but does not use ZLEMA or KST indicators.
# Removing unused functions to improve code clarity.


class IchimokuPredictControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "ichi_predict"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="3m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    # MACD parameters
    macd_fast: int = Field(
        default=22,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=36,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=17,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    # ATR parameters
    atr_length: int = Field(
        default=11,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR length: ",
            prompt_on_new=True))
    atr_multiplier: float = Field(
        default=1.5,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the ATR multiplier: ",
            prompt_on_new=True))
    # Ichimoku parameters
    tenkan_period: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Tenkan-sen period: ",
            prompt_on_new=True))
    kijun_period: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Kijun-sen period: ",
            prompt_on_new=True))
    senkou_span_b_period: int = Field(
        default=52,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Senkou Span B period: ",
            prompt_on_new=True))
    displacement: int = Field(
        default=26,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the displacement period: ",
            prompt_on_new=True))
    # Volume MA
    volume_ma_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume MA period: ",
            prompt_on_new=True))

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class IchimokuPredictController(DirectionalTradingControllerBase):

    def __init__(self, config: IchimokuPredictControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)
        
        # Initialize executors list only - remove close_types tracking at instance level
        self.executors = []

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )

        # Add Ichimoku Cloud with standard settings
        ichimoku = df.ta.ichimoku(
            tenkan=self.config.tenkan_period,
            kijun=self.config.kijun_period, 
            senkou=self.config.senkou_span_b_period,
            displacement=self.config.displacement,
            append=True
        )

        # Get the actual column names
        tenkan_col = f"ISA_{self.config.tenkan_period}"    # Conversion line
        kijun_col = f"ISB_{self.config.kijun_period}"      # Base line
        senkou_a_col = f"ITS_{self.config.tenkan_period}"  # Leading Span A
        senkou_b_col = f"IKS_{self.config.kijun_period}"   # Leading Span B
        chikou_col = f"ICS_{self.config.kijun_period}"     # Lagging Span

        # Get current values
        current_price = df['close'].iloc[-1]
        tenkan = df[tenkan_col].iloc[-1]     # Conversion line
        kijun = df[kijun_col].iloc[-1]       # Base line
        senkou_a = df[senkou_a_col].iloc[-1] # Leading Span A
        senkou_b = df[senkou_b_col].iloc[-1] # Leading Span B
        chikou = df[chikou_col].iloc[-1]     # Lagging Span

        # Calculate cloud thickness and direction
        cloud_thickness = abs(senkou_a - senkou_b)
        future_cloud_bullish = df[senkou_a_col].shift(-self.config.displacement) > df[senkou_b_col].shift(-self.config.displacement)
        
        # Calculate trend strength using cloud thickness
        avg_cloud_thickness = df[senkou_a_col].rolling(window=26).mean() - df[senkou_b_col].rolling(window=26).mean()
        cloud_strength = cloud_thickness / avg_cloud_thickness.mean()

        # Enhanced long conditions using Ichimoku predictive elements
        long_condition = (
            # Price is above the future cloud
            (current_price > max(senkou_a, senkou_b)) and
            # Future cloud is bullish (green)
            future_cloud_bullish.iloc[-1] and
            # Chikou span is above price from 26 periods ago
            (chikou > df['close'].shift(26).iloc[-1]) and
            # Tenkan-sen crosses above Kijun-sen
            (df[tenkan_col].iloc[-2] <= df[kijun_col].iloc[-2]) and
            (tenkan > kijun) and
            # Cloud thickness indicates strong trend
            (cloud_strength.iloc[-1] > 1.2) and
            # Price momentum confirmation
            (current_price > df['close'].rolling(window=20).mean().iloc[-1])
        )

        # Enhanced short conditions using Ichimoku predictive elements
        short_condition = (
            # Price is below the future cloud
            (current_price < min(senkou_a, senkou_b)) and
            # Future cloud is bearish (red)
            not future_cloud_bullish.iloc[-1] and
            # Chikou span is below price from 26 periods ago
            (chikou < df['close'].shift(26).iloc[-1]) and
            # Tenkan-sen crosses below Kijun-sen
            (df[tenkan_col].iloc[-2] >= df[kijun_col].iloc[-2]) and
            (tenkan < kijun) and
            # Cloud thickness indicates strong trend
            (cloud_strength.iloc[-1] > 1.2) and
            # Price momentum confirmation
            (current_price < df['close'].rolling(window=20).mean().iloc[-1])
        )

        # Add volume confirmation
        df[f"volume_ma_{self.config.volume_ma_period}"] = df["volume"].rolling(
            window=self.config.volume_ma_period).mean()
        volume_confirmation = df["volume"].iloc[-1] > df[f"volume_ma_{self.config.volume_ma_period}"].iloc[-1] * 1.5

        # Set signals with volume confirmation
        df["signal"] = 0
        if long_condition and volume_confirmation:
            df.loc[df.index[-1], "signal"] = 1
        elif short_condition and volume_confirmation:
            df.loc[df.index[-1], "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        
        # Configure executors based on signal
        if self.processed_data["signal"] != 0 and not self.executors:
            side = "BUY" if self.processed_data["signal"] > 0 else "SELL"
            
            # Calculate dynamic take profit and stop loss based on cloud thickness
            cloud_based_tp = float(cloud_thickness * 1.5)  # 150% of cloud thickness
            cloud_based_sl = float(cloud_thickness * 0.5)  # 50% of cloud thickness
            
            executor_config = {
                "id": str(uuid.uuid4()),  # Add unique ID
                "side": side,
                "trading_pair": self.config.trading_pair,
                "connector_name": self.config.connector_name,
                "amount": float(self.config.total_amount_quote),
                "take_profit": max(cloud_based_tp, float(self.config.take_profit)),
                "stop_loss": min(cloud_based_sl, float(self.config.stop_loss)),
                "time_limit": self.config.time_limit,
            }
            
            if self.config.trailing_stop is not None:
                executor_config["trailing_stop"] = {
                    "activation_price": float(self.config.trailing_stop.activation_price),
                    "trailing_delta": float(self.config.trailing_stop.trailing_delta)
                }
            
            self.executors.append({
                "config": executor_config,
                "close_type": None,
                "status": "ACTIVE",
                "pnl": 0.0,
                "trades": []
            })

    async def get_executors_status(self):
        """Get the status of all executors and their close types"""
        results = {
            "executors": [],
            "close_types": {
                "TAKE_PROFIT": 0,
                "STOP_LOSS": 0,
                "TIME_LIMIT": 0,
                "TRAILING_STOP": 0,
                "EARLY_STOP": 0
            },
            "accuracy_long": 0,
            "accuracy_short": 0,
            "total_executors": len(self.executors)
        }
        
        # Process each executor
        for executor in self.executors:
            # Create a copy of the executor status
            status = executor.copy()
            
            # Update close types if the executor has completed
            if status.get("close_type"):
                close_type = status["close_type"]
                if close_type in results["close_types"]:
                    results["close_types"][close_type] += 1
            
            results["executors"].append(status)
        
        # Calculate accuracies
        if self.executors:
            long_trades = [e for e in self.executors if e["config"]["side"] == "BUY"]
            short_trades = [e for e in self.executors if e["config"]["side"] == "SELL"]
            
            if long_trades:
                profitable_longs = len([t for t in long_trades if t.get("pnl", 0) > 0])
                results["accuracy_long"] = profitable_longs / len(long_trades)
            
            if short_trades:
                profitable_shorts = len([t for t in short_trades if t.get("pnl", 0) > 0])
                results["accuracy_short"] = profitable_shorts / len(short_trades)

        return results

    def on_executor_completed(self, executor_id: str, close_type: str):
        """Track executor completion"""
        for executor in self.executors:
            if executor["config"].get("id") == executor_id:
                executor["close_type"] = close_type
                break
