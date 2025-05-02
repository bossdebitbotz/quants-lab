import optuna
from optuna.storages import RDBStorage

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

# Create the connection string
storage_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Initialize storage
storage = RDBStorage(storage_url)

# Load the study
study = optuna.load_study(study_name='zlem_7', storage=storage)

# Print study information
print(f"Study name: {study.study_name}")
print(f"Number of trials: {len(study.trials)}")
print(f"Best trial value: {study.best_trial.value}")
print("Best trial parameters:")
for param_name, param_value in study.best_trial.params.items():
    print(f"  {param_name}: {param_value}") 