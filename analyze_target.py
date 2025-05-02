import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from train_improved_tft import get_db_connection, load_data

# Create output directory
import os
os.makedirs('data_analysis', exist_ok=True)

# Option 1: Query directly from database
def analyze_from_db():
    # Connect to database
    conn = get_db_connection()

    # Get recent data
    query = '''
    SELECT 
        timestamp,
        returns_10sec, 
        target_10sec as target
    FROM tft_features_10sec
    ORDER BY timestamp DESC
    LIMIT 20000;
    '''

    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} rows directly from database")
    analyze_dataset(df)

# Option 2: Use load_data function which includes preprocessing
def analyze_from_load_function():
    # Use the load_data function from train_improved_tft.py
    df = load_data(days=7, handle_missing="interpolate")
    
    # Check if successful
    if df is None:
        print("Failed to load data using load_data function")
        return
    
    print(f"Loaded {len(df)} rows using load_data function")
    # Rename target column if different
    if 'target' not in df.columns and 'target_10sec' in df.columns:
        df['target'] = df['target_10sec']
    elif 'target' not in df.columns:
        df['target'] = df['returns_10sec']
        print("Using returns_10sec as target since target column not found")
    
    analyze_dataset(df)

def analyze_dataset(df):
    """Analyze the dataset and create visualizations"""
    # Display basic statistics
    print('\nTarget Statistics:')
    print(df['target'].describe())
    print(f'Non-zero values: {(df["target"] != 0).sum()} out of {len(df)}')
    print(f'Percentage of zeros: {(df["target"] == 0).mean() * 100:.2f}%')
    
    # Check for NaN or infinite values
    print(f'NaN values: {df["target"].isna().sum()}')
    if not np.isfinite(df["target"]).all():
        print(f'Infinite values: {len(df) - np.isfinite(df["target"]).sum()}')
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['target'], bins=50, alpha=0.7)
    plt.title('Distribution of Target Values')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data_analysis/target_distribution.png')
    print('Saved target distribution plot')

    # Plot returns vs target if available
    if 'returns_10sec' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['returns_10sec'], df['target'], alpha=0.5, s=10)
        plt.title('Returns vs Target')
        plt.xlabel('returns_10sec')
        plt.ylabel('target')
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Target')
        plt.axvline(x=0, color='g', linestyle='--', label='Zero Return')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('data_analysis/returns_vs_target.png')
        print('Saved returns vs target scatter plot')
    
    # Plot target values over time
    if 'timestamp' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['target'])
        plt.title('Target Values Over Time')
        plt.xlabel('Time')
        plt.ylabel('Target Value')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('data_analysis/target_time_series.png')
        print('Saved target time series plot')
    
    # Create log-scaled histogram to see small values better
    plt.figure(figsize=(10, 6))
    # Take absolute value and add small number to avoid log(0)
    abs_target = np.abs(df['target']) + 1e-10
    plt.hist(np.log10(abs_target), bins=50, alpha=0.7)
    plt.title('Log-Scaled Distribution of Target Values')
    plt.xlabel('log10(|Target Value|)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data_analysis/target_log_distribution.png')
    print('Saved log-scaled target distribution plot')
    
    # Save distribution details to CSV for further analysis
    target_distribution = df['target'].value_counts().reset_index()
    target_distribution.columns = ['Target Value', 'Frequency']
    target_distribution = target_distribution.sort_values('Target Value')
    target_distribution.to_csv('data_analysis/target_distribution.csv', index=False)
    print('Saved target distribution to CSV')

if __name__ == "__main__":
    print("Analyzing data from database query...")
    analyze_from_db()
    
    print("\nAnalyzing data from load_data function...")
    analyze_from_load_function() 