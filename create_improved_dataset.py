import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from train_improved_tft import get_db_connection, load_data, add_derived_features

# Create output directory
os.makedirs('data_improvements', exist_ok=True)

def create_improved_dataset():
    """Create an improved dataset with better target engineering"""
    print("Loading raw data...")
    
    # Load data using existing function
    df = load_data(days=7, handle_missing="interpolate")
    
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(df)} rows of data")
    
    # Create a copy to work with
    improved_df = df.copy()
    
    # Step 1: Target engineering
    print("\nStep 1: Target Engineering")
    
    # Check if we need to add the target column
    if 'target' not in improved_df.columns and 'target_10sec' in improved_df.columns:
        improved_df['original_target'] = improved_df['target_10sec']
    elif 'target' in improved_df.columns:
        improved_df['original_target'] = improved_df['target']
    else:
        print("No target column found, using returns_10sec")
        improved_df['original_target'] = improved_df['returns_10sec']
    
    # Create various target versions
    
    # Version 1: Sign-based target (classification-like)
    improved_df['target_sign'] = np.sign(improved_df['original_target'])
    
    # Version 2: Binned magnitude target
    q_cuts = [-np.inf, -0.001, -0.0001, 0.0, 0.0001, 0.001, np.inf]
    labels = [-3, -2, -1, 1, 2, 3]
    improved_df['target_binned'] = pd.cut(improved_df['original_target'], 
                                         bins=q_cuts, 
                                         labels=labels)
    improved_df['target_binned'] = improved_df['target_binned'].astype(float)
    
    # Version 3: Log-scaled target to emphasize small changes
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    sign_target = np.sign(improved_df['original_target'])
    log_abs_target = np.log10(np.abs(improved_df['original_target']) + epsilon)
    improved_df['target_log'] = sign_target * log_abs_target
    
    # Version 4: Moving average adjusted target
    # Subtract recent average to focus on relative changes
    ma_window = 30  # 5 minutes (30 x 10-sec bars)
    recent_avg = improved_df['original_target'].rolling(window=ma_window).mean().fillna(0)
    improved_df['target_ma_adjusted'] = improved_df['original_target'] - recent_avg
    
    # Step 2: Filter out periods with no activity
    print("\nStep 2: Filtering inactive periods")
    original_rows = len(improved_df)
    
    # Check price movement
    price_change = np.abs(improved_df['close'].pct_change())
    price_movement = price_change.rolling(window=30).sum()
    
    # Keep rows with some price movement or non-zero targets
    active_periods = ((price_movement > 0) | (improved_df['original_target'] != 0))
    improved_df = improved_df[active_periods]
    
    print(f"Filtered from {original_rows} to {len(improved_df)} rows with price activity")
    
    # Step 3: Add more engineered features
    print("\nStep 3: Adding engineered features")
    
    # Add target lags as features
    for lag in [1, 2, 3, 5, 10]:
        improved_df[f'target_lag_{lag}'] = improved_df['original_target'].shift(lag)
    
    # Add rolling statistics of past returns
    for window in [5, 10, 30]:
        # Rolling mean (trend)
        improved_df[f'returns_mean_{window}'] = improved_df['returns_10sec'].rolling(window=window).mean()
        # Rolling std (volatility)
        improved_df[f'returns_std_{window}'] = improved_df['returns_10sec'].rolling(window=window).std()
        # Rolling sum of absolute returns (activity)
        improved_df[f'returns_abs_sum_{window}'] = np.abs(improved_df['returns_10sec']).rolling(window=window).sum()
    
    # Add imbalance momentum
    improved_df['imbalance_momentum'] = improved_df['imbalance'].diff().rolling(window=5).mean()
    
    # Fill all NaN values
    improved_df = improved_df.fillna(0)
    
    # Save the improved dataset
    print("\nSaving improved dataset...")
    improved_df.to_csv('data_improvements/improved_dataset.csv', index=True)
    
    # Create a smaller version for faster experimentation
    small_df = improved_df.sample(min(20000, len(improved_df)), random_state=42)
    small_df.to_csv('data_improvements/improved_dataset_small.csv', index=True)
    
    # Compare the target distributions
    plot_target_comparison(improved_df)
    
    print(f"\nImproved dataset created with {len(improved_df)} rows and saved to data_improvements/improved_dataset.csv")
    print(f"Small dataset created with {len(small_df)} rows and saved to data_improvements/improved_dataset_small.csv")
    
    return improved_df

def plot_target_comparison(df):
    """Plot comparison of different target versions"""
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Original target histogram
    plt.subplot(2, 2, 1)
    plt.hist(df['original_target'], bins=50, alpha=0.7)
    plt.title('Original Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Sign-based target
    plt.subplot(2, 2, 2)
    sign_counts = df['target_sign'].value_counts()
    plt.bar(sign_counts.index, sign_counts.values, alpha=0.7)
    plt.title('Sign-Based Target Distribution')
    plt.xlabel('Sign (-1, 0, 1)')
    plt.ylabel('Frequency')
    plt.xticks([-1, 0, 1], ['Negative', 'Zero', 'Positive'])
    plt.grid(True)
    
    # Binned magnitude target
    plt.subplot(2, 2, 3)
    bin_counts = df['target_binned'].value_counts().sort_index()
    plt.bar(bin_counts.index, bin_counts.values, alpha=0.7)
    plt.title('Binned Magnitude Target Distribution')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Log-scaled target
    plt.subplot(2, 2, 4)
    plt.hist(df['target_log'], bins=50, alpha=0.7)
    plt.title('Log-Scaled Target Distribution')
    plt.xlabel('Log Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_improvements/target_comparisons.png')
    print("Target comparison plot saved to data_improvements/target_comparisons.png")
    
    # Create additional plots for MA-adjusted target
    plt.figure(figsize=(10, 6))
    plt.hist(df['target_ma_adjusted'], bins=50, alpha=0.7)
    plt.title('Moving Average Adjusted Target Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data_improvements/target_ma_adjusted.png')
    print("MA-adjusted target plot saved to data_improvements/target_ma_adjusted.png")
    
    # Create scatterplot comparing original vs log-scaled target
    plt.figure(figsize=(10, 6))
    plt.scatter(df['original_target'], df['target_log'], alpha=0.5, s=5)
    plt.title('Original vs Log-Scaled Target')
    plt.xlabel('Original Target')
    plt.ylabel('Log-Scaled Target')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data_improvements/original_vs_log_target.png')
    print("Original vs log target plot saved to data_improvements/original_vs_log_target.png")

if __name__ == "__main__":
    print("Creating improved dataset with better target engineering...")
    improved_df = create_improved_dataset()
    
    # Print some statistics
    print("\nTarget Statistics:")
    print("\nOriginal Target:")
    print(improved_df['original_target'].describe())
    
    print("\nSign-Based Target:")
    print(improved_df['target_sign'].value_counts())
    
    print("\nBinned Magnitude Target:")
    print(improved_df['target_binned'].value_counts().sort_index())
    
    print("\nLog-Scaled Target:")
    print(improved_df['target_log'].describe()) 