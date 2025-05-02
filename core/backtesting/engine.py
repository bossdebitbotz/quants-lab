import logging
import os
from typing import Dict, Optional

import pandas as pd

from core.data_structures.backtesting_result import BacktestingResult
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    def __init__(self, load_cached_data: bool = True, root_path: str = "", custom_backtester: Optional[BacktestingEngineBase] = None):
        self._bt_engine = custom_backtester if custom_backtester is not None else BacktestingEngineBase()
        self.root_path = root_path
        if load_cached_data:
            self._load_candles_cache(root_path)

    async def cleanup(self):
        """
        Clean up resources and connections.
        """
        try:
            # Clear any cached data
            if hasattr(self._bt_engine, 'backtesting_data_provider'):
                self._bt_engine.backtesting_data_provider.candles_feeds.clear()
            # Add any other cleanup tasks here
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _load_candles_cache(self, root_path: str):
        cache_dir = os.path.join(root_path, "data", "candles")
        print(f"Attempting to load cache from directory: {cache_dir}")
        if not os.path.exists(cache_dir):
            print(f"Cache directory not found: {cache_dir}")
            return

        all_files = os.listdir(cache_dir)
        print(f"Found files in cache directory: {all_files}")
        for file in all_files:
            if file == ".gitignore" or not file.endswith(".parquet"):
                continue
            file_path = os.path.join(cache_dir, file)
            print(f"Processing cache file: {file}")
            try:
                parts = file.split('.')[0].split('|')
                if len(parts) != 3:
                    print(f"  Skipping file with unexpected name format: {file}")
                    continue
                connector_name, trading_pair, interval = parts
                print(f"  Parsed components: connector={connector_name}, pair={trading_pair}, interval={interval}")

                candles = pd.read_parquet(file_path)
                print(f"  Successfully read Parquet file. Shape: {candles.shape}")

                if 'timestamp' not in candles.columns and not isinstance(candles.index, pd.DatetimeIndex):
                     print(f"  Skipping file: Neither 'timestamp' column nor DatetimeIndex found in {file}")
                     continue

                # Handle case where timestamp might be index AND a column
                if isinstance(candles.index, pd.DatetimeIndex):
                     print("  Index is DatetimeIndex.")
                     # If a 'timestamp' column ALSO exists, drop it to avoid conflict, preferring the index
                     if 'timestamp' in candles.columns:
                         print("  Warning: Both DatetimeIndex and 'timestamp' column exist. Dropping column, using index.")
                         candles.drop(columns=['timestamp'], inplace=True)

                     # Now, reset the index. The timestamp data from the index will become a column.
                     candles.reset_index(inplace=True)
                     # Rename the column created from the index (likely 'index' or 'timestamp_dt') to 'timestamp'
                     if 'index' in candles.columns:
                         candles.rename(columns={'index': 'timestamp'}, inplace=True)
                         print("  Renamed reset index column 'index' to 'timestamp'.")
                     elif 'timestamp_dt' in candles.columns: # If index was named 'timestamp_dt'
                         candles.rename(columns={'timestamp_dt': 'timestamp'}, inplace=True)
                         print("  Renamed reset index column 'timestamp_dt' to 'timestamp'.")
                     elif 'timestamp' in candles.columns: # If the index was already named 'timestamp'
                         print("  Index was already named 'timestamp', column retained after reset.")
                         pass # Column already has the correct name
                     else:
                         print(f"  Error: Could not find column derived from index after reset for {file}")
                         continue # Skip this file

                # If index was not DatetimeIndex, we rely on the existing 'timestamp' column (checked at the start)
                # Ensure the 'timestamp' column is now present after potential index reset/rename
                elif 'timestamp' not in candles.columns:
                     print(f"  Skipping file: 'timestamp' column not found after processing index for {file}")
                     continue

                # --- Timestamp processing: Ensure 'timestamp' column is numeric seconds ---
                if not pd.api.types.is_numeric_dtype(candles['timestamp']):
                    # If it's datetime (e.g., from renaming index), convert back to numeric seconds
                    if pd.api.types.is_datetime64_any_dtype(candles['timestamp']):
                        print("  Converting 'timestamp' column from datetime back to numeric seconds.")
                        timestamp_col = candles['timestamp']
                        # Ensure the column is timezone-aware (localize to UTC if naive)
                        if timestamp_col.dt.tz is None:
                            print("  Timestamp column is tz-naive. Localizing to UTC.")
                            timestamp_col = timestamp_col.dt.tz_localize('UTC')
                        elif timestamp_col.dt.tz != 'UTC':
                             # Should ideally not happen if data source is consistent, but handle anyway
                             print(f"  Timestamp column is tz-aware but not UTC ({timestamp_col.dt.tz}). Converting to UTC.")
                             timestamp_col = timestamp_col.dt.tz_convert('UTC')
                        
                        # Now perform subtraction with UTC epoch
                        candles['timestamp'] = (timestamp_col - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
                    else:
                        # If it's some other non-numeric type, skip
                        print(f"  Skipping file: Timestamp column in {file} is not numeric or datetime after processing index.")
                        continue

                # Check for ms vs s and convert if necessary
                if candles['timestamp'].max() > 1e12:
                     print(f"  Converting timestamp column from ms to s for {file}")
                     candles['timestamp'] = candles['timestamp'] // 1000

                # --- Create the DatetimeIndex for Hummingbot internal use ---
                candles['timestamp_dt'] = pd.to_datetime(candles['timestamp'], unit='s', errors='coerce')
                candles.dropna(subset=['timestamp_dt'], inplace=True)
                candles.set_index('timestamp_dt', inplace=True)
                candles.index.name = None

                columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                           'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                for col in columns:
                    if col in candles.columns:
                        candles[col] = pd.to_numeric(candles[col], errors='coerce')
                    else:
                        print(f"  Warning: Expected column '{col}' not found in {file}.")

                essential_cols = ['open', 'high', 'low', 'close', 'volume']
                candles.dropna(subset=essential_cols, inplace=True)

                if candles.empty:
                    print(f"  Skipping file: DataFrame empty after processing for {file}")
                    continue

                cache_key = f"{connector_name}_{trading_pair}_{interval}"
                self._bt_engine.backtesting_data_provider.candles_feeds[cache_key] = candles

                start_time_ts = candles.index.min().timestamp()
                end_time_ts = candles.index.max().timestamp()
                start_time_dt = candles.index.min()
                end_time_dt = candles.index.max()

                self._bt_engine.backtesting_data_provider.start_time = int(start_time_ts)
                self._bt_engine.backtesting_data_provider.end_time = int(end_time_ts)
                print(f"  Successfully loaded cache for key: {cache_key}")
                print(f"  Cache data range: {start_time_dt} to {end_time_dt}")

            except Exception as e:
                logger.error(f"Error loading cache file {file}: {e}")
                import traceback
                traceback.print_exc()

    def load_candles_cache_by_connector_pair(self, connector_name: str, trading_pair: str, root_path: str = ""):
            all_files = os.listdir(os.path.join(root_path, "data", "candles"))
            for file in all_files:
                if file == ".gitignore":
                    continue
                try:
                    if connector_name in file and trading_pair in file:
                        connector_name, trading_pair, interval = file.split(".")[0].split("|")
                        candles = pd.read_parquet(os.path.join(root_path, "data", "candles", file))
                        candles.index = pd.to_datetime(candles.timestamp, unit='s')
                        candles.index.name = None
                        columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                                   'n_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
                        for column in columns:
                            candles[column] = pd.to_numeric(candles[column])
                        self._bt_engine.backtesting_data_provider.candles_feeds[
                            f"{connector_name}_{trading_pair}_{interval}"] = candles
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")

    def get_controller_config_instance_from_dict(self, config: Dict):
        return BacktestingEngineBase.get_controller_config_instance_from_dict(
            config_data=config,
            controllers_module="controllers",
        )

    async def run_backtesting(self, config: ControllerConfigBase, start: int,
                              end: int, backtesting_resolution: str, trade_cost: float = 0.0006) -> BacktestingResult:
        bt_result = await self._bt_engine.run_backtesting(config, start, end, backtesting_resolution, trade_cost)
        return BacktestingResult(bt_result, config)

    async def backtest_controller_from_yml(self,
                                           config_file: str,
                                           controllers_conf_dir_path: str,
                                           start: int,
                                           end: int,
                                           backtesting_resolution: str = "1m",
                                           trade_cost: float = 0.0006,
                                           backtester: Optional[BacktestingEngineBase] = None):
        config = self._bt_engine.get_controller_config_instance_from_yml(config_file, controllers_conf_dir_path)
        return await self.run_backtesting(config, start, end, backtesting_resolution, trade_cost, backtester)
