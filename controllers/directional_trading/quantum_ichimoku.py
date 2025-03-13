from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, validator
import aiohttp
import pandas as pd
import asyncio
import contextlib


class QuantumIchimokuControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = "quantum_ichimoku"
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
    ema_short: int = Field(
        default=8,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA short period: ",
            prompt_on_new=True))
    ema_medium: int = Field(
        default=29,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA medium period: ",
            prompt_on_new=True))
    ema_long: int = Field(
        default=31,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the EMA long period: ",
            prompt_on_new=True))
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
    volume_ma_period: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the volume MA period: ",
            prompt_on_new=True))
    
    # Advanced Momentum Settings
    rsi_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the RSI period: ",
            prompt_on_new=True))
    rsi_overbought: int = Field(
        default=70,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter RSI overbought level: ",
            prompt_on_new=True))
    rsi_oversold: int = Field(
        default=30,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter RSI oversold level: ",
            prompt_on_new=True))
            
    # Volatility Settings
    bb_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter Bollinger Bands period: ",
            prompt_on_new=True))
    bb_std: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter Bollinger Bands standard deviation: ",
            prompt_on_new=True))
            
    # Volume Profile Settings
    vwap_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter VWAP period: ",
            prompt_on_new=True))
            
    # Trend Strength Settings
    adx_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ADX period: ",
            prompt_on_new=True))
    adx_threshold: int = Field(
        default=25,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ADX threshold: ",
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


class QuantumIchimokuController(DirectionalTradingControllerBase):

    def __init__(self, config: QuantumIchimokuControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        self._client_session = None
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def cleanup(self):
        """Cleanup resources"""
        if self._client_session:
            with contextlib.suppress(Exception):
                await self._client_session.close()
            self._client_session = None

    async def update_processed_data(self):
        try:
            df = self.market_data_provider.get_candles_df(
                connector_name=self.config.candles_connector,
                trading_pair=self.config.candles_trading_pair,
                interval=self.config.interval,
                max_records=self.max_records
            )
            
            if df is None or df.empty:
                logger.error("No data received from market data provider")
                return
            
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, unit='s')
                
            # Calculate volume MA first
            df[f"volume_ma_{self.config.volume_ma_period}"] = df["volume"].rolling(
                window=self.config.volume_ma_period).mean()
                
            # Custom VWAP calculation instead of using pandas_ta
            def calculate_vwap(df):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                return vwap
                
            # Replace the pandas_ta VWAP with our custom calculation
            df['VWAP'] = calculate_vwap(df)
            
            # Add indicators with error handling
            try:
                # MACD
                macd_columns = df.ta.macd(
                    fast=self.config.macd_fast,
                    slow=self.config.macd_slow,
                    signal=self.config.macd_signal,
                    append=True
                )
                
                # ATR
                atr_columns = df.ta.atr(length=self.config.atr_length, append=True)
                
                # EMAs
                ema_short_columns = df.ta.ema(length=self.config.ema_short, append=True)
                ema_medium_columns = df.ta.ema(length=self.config.ema_medium, append=True)
                ema_long_columns = df.ta.ema(length=self.config.ema_long, append=True)

                # Ichimoku with error handling
                ichimoku = df.ta.ichimoku(
                    tenkan=self.config.tenkan_period,
                    kijun=self.config.kijun_period,
                    senkou=self.config.senkou_span_b_period,
                    displacement=self.config.displacement,
                    append=True
                )
                
                
                # Safely get Ichimoku values
                def safe_get_column(df, col_name, default=None):
                    return df[col_name] if col_name in df.columns else default
                
                # Map Ichimoku values with fallbacks
                isa = safe_get_column(df, f"ISA_{self.config.tenkan_period}")
                isb = safe_get_column(df, f"ISB_{self.config.kijun_period}")
                its = safe_get_column(df, f"ITS_{self.config.tenkan_period}")
                iks = safe_get_column(df, f"IKS_{self.config.kijun_period}")
                
                if any(x is None for x in [isa, isb, its, iks]):
                    logger.error("Missing required Ichimoku columns")
                    return
                
                # Calculate volume ratio using the already calculated volume MA
                volume = df["volume"]
                volume_ma = df[f"volume_ma_{self.config.volume_ma_period}"]
                volume_ratio = volume / volume_ma

                # Advanced Indicators
                # RSI
                df.ta.rsi(length=self.config.rsi_period, append=True)
                
                # Bollinger Bands
                df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
                
                # ADX
                df.ta.adx(length=self.config.adx_length, append=True)
                
                # Enhanced Volume Analysis
                df["volume_ema"] = df.ta.ema(length=self.config.volume_ma_period, close=df["volume"])
                df["volume_std"] = df["volume"].rolling(window=self.config.volume_ma_period).std()
                df["volume_z_score"] = (df["volume"] - df["volume_ema"]) / df["volume_std"]

                # Volatility Regime
                df["atr_volatility"] = df[f"ATRr_{self.config.atr_length}"] / df["close"] * 100
                df["volatility_regime"] = df["atr_volatility"].rolling(window=20).mean()

                # Advanced Signal Generation
                long_condition = (
                    # Trend Alignment
                    (df["close"] > df[f"EMA_{self.config.ema_short}"]) &
                    (df[f"EMA_{self.config.ema_short}"] > df[f"EMA_{self.config.ema_medium}"]) &
                    (df[f"EMA_{self.config.ema_medium}"] > df[f"EMA_{self.config.ema_long}"]) &
                    
                    # Ichimoku Confirmation - Using correct column names
                    (df["close"] > isa) &  # Price above Tenkan-sen
                    (df["close"] > isb) &  # Price above Kijun-sen
                    (its > iks) &          # Senkou Span A above Senkou Span B
                    
                    # Momentum Confirmation
                    (df[f"RSI_{self.config.rsi_period}"] > 50) &
                    (df[f"RSI_{self.config.rsi_period}"] < self.config.rsi_overbought) &
                    
                    # Volume Confirmation
                    (df["volume_z_score"] > 1.0) &
                    
                    # Trend Strength
                    (df[f"ADX_{self.config.adx_length}"] > self.config.adx_threshold) &
                    
                    # Volatility Filter
                    (df["volatility_regime"] < df["volatility_regime"].rolling(window=100).mean() * 1.5)
                )

                short_condition = (
                    # Trend Alignment
                    (df["close"] < df[f"EMA_{self.config.ema_short}"]) &
                    (df[f"EMA_{self.config.ema_short}"] < df[f"EMA_{self.config.ema_medium}"]) &
                    (df[f"EMA_{self.config.ema_medium}"] < df[f"EMA_{self.config.ema_long}"]) &
                    
                    # Ichimoku Confirmation - Using correct column names
                    (df["close"] < isa) &  # Price below Tenkan-sen
                    (df["close"] < isb) &  # Price below Kijun-sen
                    (its < iks) &          # Senkou Span A below Senkou Span B
                    
                    # Momentum Confirmation
                    (df[f"RSI_{self.config.rsi_period}"] < 50) &
                    (df[f"RSI_{self.config.rsi_period}"] > self.config.rsi_oversold) &
                    
                    # Volume Confirmation
                    (df["volume_z_score"] > 1.0) &
                    
                    # Trend Strength
                    (df[f"ADX_{self.config.adx_length}"] > self.config.adx_threshold) &
                    
                    # Volatility Filter
                    (df["volatility_regime"] < df["volatility_regime"].rolling(window=100).mean() * 1.5)
                )

                # Signal Generation
                df["signal"] = 0
                df.loc[long_condition, "signal"] = 1
                df.loc[short_condition, "signal"] = -1

                # Position Sizing based on Volatility
                df["position_size_factor"] = 1.0 / df["atr_volatility"]
                df["position_size_factor"] = df["position_size_factor"].clip(0.25, 1.0)

                # Update processed data
                self.processed_data["signal"] = df["signal"].iloc[-1]
                self.processed_data["position_size"] = df["position_size_factor"].iloc[-1]
                self.processed_data["features"] = df
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                raise
                
        except asyncio.CancelledError:
            logger.warning("Operation was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in update_processed_data: {str(e)}")
            raise


class QuantumIchimokuTrader:
    def __init__(self):
        self.session = None
        self.connector = None
        self.api_url = None  # Add API URL if needed
    
    async def create_session(self):
        # Create new session if none exists
        if not self.session:
            self.connector = aiohttp.TCPConnector(limit=100)
            self.session = aiohttp.ClientSession(connector=self.connector)
        return self.session

    async def cleanup(self):
        # Properly close session and connector
        if self.session:
            await self.session.close()
            self.session = None
        if self.connector:
            await self.connector.close()
            self.connector = None

    async def __aenter__(self):
        await self.create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def fetch_data(self):
        try:
            async with self.session.get(self.api_url) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def preprocess_vwap_data(self, price_series, volume_series):
        # Sort price and volume data by datetime
        df = pd.DataFrame({
            'price': price_series,
            'volume': volume_series
        })
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        return df['price'], df['volume']

    async def calculate_vwap(self, price_series, volume_series):
        # Preprocess and sort data first
        price_series, volume_series = self.preprocess_vwap_data(price_series, volume_series)
        
        # Continue with VWAP calculation
        typical_price = price_series
        vwap = (typical_price * volume_series).cumsum() / volume_series.cumsum()
        return vwap

    @staticmethod
    def calculate_vwap_anchored(df, anchor='D'):
        """Calculate VWAP with period anchoring"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df['date'] = df.index.date
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        vwap = df.groupby('date').apply(
            lambda x: (
                (x['high'] + x['low'] + x['close']) / 3 * x['volume']
            ).cumsum() / x['volume'].cumsum()
        )
        return vwap

    def process_data(self, df):
        """Process data and add VWAP"""
        if df is not None:
            df['VWAP'] = self.calculate_vwap_anchored(df)
        return df
