import asyncio
import datetime
from decimal import Decimal

from elite_oro_config_gen import EliteOroConfigGenerator
from controllers.directional_trading.elite_oro_1 import EliteOroController


async def test_elite_oro_config():
    # Define start and end dates for the test
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 1, 31)
    
    # Create the config generator
    config_generator = EliteOroConfigGenerator(start_date, end_date)
    
    # Create a mock trial object with fixed values for testing
    class MockTrial:
        def suggest_int(self, name, low, high):
            # Return middle values for testing
            if name == "zlema_fast":
                return 5
            elif name == "zlema_medium":
                return 15
            elif name == "zlema_slow":
                return 25
            elif name == "atr_length":
                return 15
            elif name == "volume_ma_period":
                return 20
            elif name == "price_channel_period":
                return 10
            elif name == "rsi_period":
                return 14
            elif name == "max_executors_per_side":
                return 2
            return (low + high) // 2
        
        def suggest_float(self, name, low, high, step=None):
            # Return middle values for testing
            if name == "atr_multiplier":
                return 1.2
            elif name == "volume_surge":
                return 2.0
            elif name == "take_profit":
                return 0.29
            elif name == "stop_loss":
                return 0.06
            elif name == "trailing_stop_activation_price":
                return 0.025
            elif name == "trailing_stop_trailing_delta":
                return 0.019
            return (low + high) / 2
    
    # Generate the config
    backtest_config = await config_generator.generate_config(MockTrial())
    
    # Print the config details
    print(f"Generated config ID: {backtest_config.config.id}")
    print(f"Trading pair: {backtest_config.config.trading_pair}")
    print(f"Connector: {backtest_config.config.connector_name}")
    print(f"Interval: {backtest_config.config.interval}")
    print(f"ZLEMA parameters: fast={backtest_config.config.zlema_fast}, medium={backtest_config.config.zlema_medium}, slow={backtest_config.config.zlema_slow}")
    print(f"ATR parameters: length={backtest_config.config.atr_length}, multiplier={backtest_config.config.atr_multiplier}")
    print(f"Volume parameters: ma_period={backtest_config.config.volume_ma_period}, surge={backtest_config.config.volume_surge}")
    print(f"Price channel period: {backtest_config.config.price_channel_period}")
    print(f"RSI period: {backtest_config.config.rsi_period}")
    print(f"Take profit: {backtest_config.config.take_profit}")
    print(f"Stop loss: {backtest_config.config.stop_loss}")
    print(f"Trailing stop: activation_price={backtest_config.config.trailing_stop.activation_price}, trailing_delta={backtest_config.config.trailing_stop.trailing_delta}")
    print(f"Max executors per side: {backtest_config.config.max_executors_per_side}")
    print(f"Time limit: {backtest_config.config.time_limit}")
    print(f"Cooldown time: {backtest_config.config.cooldown_time}")
    
    # Try to create a controller with the config
    try:
        controller = EliteOroController(backtest_config.config)
        print("\nSuccessfully created EliteOroController with the generated config!")
    except Exception as e:
        print(f"\nError creating controller: {e}")


if __name__ == "__main__":
    asyncio.run(test_elite_oro_config()) 