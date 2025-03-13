import asyncio
import datetime
import logging
import os
from decimal import Decimal

import optuna
from optuna.samplers import TPESampler

from elite_oro_config_gen import EliteOroConfigGenerator
from core.backtesting.engine import BacktestingEngine
from core.backtesting.optimizer import BacktestingOptimizer


async def optimize_elite_oro():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Define the optimization period
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 3, 31)
    
    # Create the config generator
    config_generator = EliteOroConfigGenerator(start_date, end_date)
    
    # Define the optimization objective
    async def objective(trial):
        # Generate a configuration for this trial
        config = await config_generator.generate_config(trial)
        
        # Create a backtesting engine
        engine = BacktestingEngine(config)
        
        # Run the backtest
        result = await engine.run()
        
        # Calculate the objective value (profit and loss)
        pnl = result.get_pnl()
        
        # Log the result
        logger.info(f"Trial {trial.number}: PnL = {pnl}")
        
        # Return the objective value (negative for maximization)
        return -pnl
    
    # Create the optimizer
    optimizer = BacktestingOptimizer(
        objective=objective,
        n_trials=50,  # Number of trials to run
        sampler=TPESampler(seed=42),  # Use TPE sampler with a fixed seed for reproducibility
    )
    
    # Run the optimization
    study = await optimizer.optimize()
    
    # Print the best parameters
    logger.info("Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Print the best value
    logger.info(f"Best PnL: {-study.best_value}")
    
    # Save the study
    os.makedirs("optimization_results", exist_ok=True)
    study.trials_dataframe().to_csv("optimization_results/elite_oro_optimization.csv")
    
    return study


if __name__ == "__main__":
    asyncio.run(optimize_elite_oro()) 