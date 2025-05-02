import os
import logging
import argparse
import pandas as pd
import numpy as np
import psycopg2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime, timedelta
from models.improved_tft import TemporalFusionTransformer, create_improved_data_loaders
from sklearn.preprocessing import StandardScaler
import json
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from scipy.stats import spearmanr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tft_improved_trainer")

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5438,
    'user': 'backtest_user',
    'password': 'backtest_password',
    'database': 'backtest_db'
}

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {type(e).__name__} - {str(e)}")
        return None

def load_data(days=14, handle_missing="fill_zero"):
    """
    Load and prepare 10-second bar data from the database
    
    Parameters:
    days (int): Number of days of data to retrieve
    handle_missing (str): Strategy for handling missing values:
                         - "fill_zero": Replace NaNs with zeros
                         - "fill_mean": Replace NaNs with column means
                         - "fill_forward": Forward fill missing values
                         - "interpolate": Linear interpolation
    """
    try:
        conn = get_db_connection()
        
        # Get the latest date in the database
        date_query = """
        SELECT MAX(timestamp) FROM tft_features_10sec
        """
        with conn.cursor() as cursor:
            cursor.execute(date_query)
            latest_date = cursor.fetchone()[0]
        
        if latest_date is None:
            logger.error("No data found in tft_features_10sec table")
            return None
        
        # Calculate the date n days ago from the latest date
        start_date = latest_date - timedelta(days=days)
        
        logger.info(f"Fetching data from {start_date} to {latest_date}")
        
        # Query to get all features from 10-second bars
        query = """
        SELECT 
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            bid_vol,
            ask_vol,
            imbalance,
            spread_mean,
            spread_pct_mean,
            new_bid_orders,
            new_ask_orders,
            canceled_bid_orders,
            canceled_ask_orders,
            executed_bid_orders,
            executed_ask_orders,
            buy_sell_imbalance,
            returns_10sec,
            returns_30sec,
            returns_1min,
            volatility_1min,
            target_10sec as target
        FROM tft_features_10sec
        WHERE timestamp >= %s
        ORDER BY timestamp
        """
        
        # Read data into DataFrame
        df = pd.read_sql_query(query, conn, params=[start_date])
        
        logger.info(f"Retrieved {len(df)} rows of raw data")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values in columns: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Handle missing values based on the specified strategy
        if handle_missing == "fill_zero":
            df = df.fillna(0)
            logger.info("Missing values filled with zeros")
        elif handle_missing == "fill_mean":
            # Calculate means only on numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns
            means = df[numeric_cols].mean()
            df[numeric_cols] = df[numeric_cols].fillna(means)
            logger.info("Missing values filled with column means")
        elif handle_missing == "fill_forward":
            df = df.fillna(method='ffill')
            # Any remaining NaNs (e.g., at start) filled with zeros
            df = df.fillna(0)
            logger.info("Missing values forward-filled")
        elif handle_missing == "interpolate":
            df = df.interpolate(method='linear', limit_direction='both')
            # Any remaining NaNs filled with zeros
            df = df.fillna(0)
            logger.info("Missing values interpolated linearly")
        else:
            logger.warning(f"Unknown missing value strategy: {handle_missing}, defaulting to fill_zero")
            df = df.fillna(0)
        
        # Generate additional features
        df = add_derived_features(df)
        
        # Set timestamp as index for convenience
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        if conn:
            conn.close()

def add_derived_features(df):
    """
    Add derived features to enhance the model's predictive power
    with advanced microstructure metrics
    """
    logger.info("Adding derived features and microstructure metrics...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    try:
        # Fix zero std returns if they exist
        for col in ['returns_10sec', 'returns_30sec', 'returns_1min']:
            if df[col].std() == 0 or df[col].isnull().all():
                # Calculate returns properly if the column data is invalid
                if col == 'returns_10sec':
                    df[col] = df['close'].pct_change()
                elif col == 'returns_30sec':
                    df[col] = df['close'].pct_change(3)  # 3 * 10sec = 30sec
                elif col == 'returns_1min':
                    df[col] = df['close'].pct_change(6)  # 6 * 10sec = 1min
                logger.info(f"Recalculated {col}")
    
        # Fix volatility calculation if it has zero std
        if df['volatility_1min'].std() == 0 or df['volatility_1min'].isnull().all():
            # Calculate rolling standard deviation of returns
            df['volatility_1min'] = df['returns_10sec'].rolling(window=6).std().fillna(0)
            logger.info(f"Recalculated volatility_1min")
        
        # ===== Basic Technical Features =====
        # Trend indicator (simple moving average crossover)
        df['sma_fast'] = df['close'].rolling(window=6).mean()  # 1-minute SMA
        df['sma_slow'] = df['close'].rolling(window=18).mean()  # 3-minute SMA
        df['trend_indicator'] = (df['sma_fast'] > df['sma_slow']).astype(float)
        
        # Volatility regime (high/low volatility periods)
        vol_median = df['volatility_1min'].median()
        df['high_volatility'] = (df['volatility_1min'] > vol_median).astype(float)
        
        # Price momentum features
        # RSI-like indicator for 10-second bars (simplified calculation)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ===== ADVANCED MICROSTRUCTURE FEATURES =====
        
        # ===== 1. Order Flow Imbalance Ratios =====
        # Normalized order flow imbalance
        if 'bid_vol' in df.columns and 'ask_vol' in df.columns:
            total_vol = df['bid_vol'].fillna(0) + df['ask_vol'].fillna(0)
            total_vol = total_vol.replace(0, 1e-9)  # Avoid division by zero
            df['norm_imbalance'] = (df['bid_vol'].fillna(0) - df['ask_vol'].fillna(0)) / total_vol
        
        # Order flow acceleration (rate of change in imbalance)
        df['imbalance_change'] = df['imbalance'].diff()
        df['imbalance_acceleration'] = df['imbalance_change'].diff()
        
        # Cumulative imbalance over multiple timeframes
        df['cum_imbalance_1min'] = df['imbalance'].rolling(window=6).sum()
        df['cum_imbalance_3min'] = df['imbalance'].rolling(window=18).sum()
        
        # Relative strength of buy/sell pressure
        if 'new_bid_orders' in df.columns and 'new_ask_orders' in df.columns:
            df['new_orders_ratio'] = df['new_bid_orders'].fillna(0) / (df['new_ask_orders'].fillna(0) + 1e-9)
            # Log transform to make more symmetric around 0
            df['log_order_ratio'] = np.log(df['new_orders_ratio'].replace(0, 1e-9))
        
        # ===== 2. Limit Order Book Pressure Metrics =====
        # Order cancellation ratios (market sentiment)
        if all(col in df.columns for col in ['canceled_bid_orders', 'canceled_ask_orders', 'new_bid_orders', 'new_ask_orders']):
            # Cancel-to-new order ratio (separate for bid and ask)
            df['bid_cancel_ratio'] = df['canceled_bid_orders'].fillna(0) / (df['new_bid_orders'].fillna(0) + 1e-9)
            df['ask_cancel_ratio'] = df['canceled_ask_orders'].fillna(0) / (df['new_ask_orders'].fillna(0) + 1e-9)
            
            # Relative cancel rates (bull/bear signal)
            df['relative_cancel_ratio'] = df['bid_cancel_ratio'] / (df['ask_cancel_ratio'] + 1e-9)
            df['log_cancel_ratio'] = np.log(df['relative_cancel_ratio'].replace(0, 1e-9))
        
        # Order book pressure metrics
        if 'spread_pct_mean' in df.columns:
            # Effective spread impact - wider spreads indicate lower liquidity
            df['inv_liquidity'] = df['spread_pct_mean'] * df['volatility_1min']
            
            # Spread changes - sudden increases signal volatility events
            df['spread_change'] = df['spread_pct_mean'].pct_change()
            df['spread_z_score'] = (df['spread_pct_mean'] - df['spread_pct_mean'].rolling(window=60).mean()) / df['spread_pct_mean'].rolling(window=60).std().replace(0, 1e-9)
        
        # ===== 3. Trade Impact Metrics =====
        # Price impact of trades
        if 'executed_bid_orders' in df.columns and 'executed_ask_orders' in df.columns:
            # Create proxy for price impact of executed orders
            df['executed_orders_imbalance'] = df['executed_bid_orders'].fillna(0) - df['executed_ask_orders'].fillna(0)
            
            # Normalized execution imbalance
            total_executed = df['executed_bid_orders'].fillna(0) + df['executed_ask_orders'].fillna(0)
            df['normalized_exec_imbalance'] = df['executed_orders_imbalance'] / (total_executed + 1e-9)
            
            # Market impact proxies
            df['returns_per_volume'] = df['returns_10sec'] / (df['volume'].fillna(0) + 1e-9)
            
            # Trade impact on order book (depletion rate)
            if 'volume' in df.columns:
                df['market_impact'] = df['returns_10sec'].abs() / (df['volume'].fillna(0) + 1e-9)

        # ===== 4. Non-linear and Interaction Features =====
        # Interaction between volatility and imbalance (high volatility + imbalance signals strong moves)
        df['vol_imbalance_interaction'] = df['volatility_1min'] * df['imbalance'].abs()
        
        # Sign consistency across timeframes (trend strength)
        df['sign_consistency'] = ((np.sign(df['returns_10sec']) == np.sign(df['returns_30sec'])) & 
                                 (np.sign(df['returns_30sec']) == np.sign(df['returns_1min']))).astype(float)
        
        # Microstructure trend - order flow aligned with price changes
        if 'imbalance' in df.columns:
            df['aligned_flow'] = (np.sign(df['imbalance']) == np.sign(df['returns_10sec'])).astype(float)
            # Rolling sum to get cumulative alignment
            df['aligned_flow_cumul'] = df['aligned_flow'].rolling(window=10).sum() / 10
        
        # ===== 5. Time-based Contextual Features =====
        # Extract hour information (market activity varies by hour)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            
            # Add cyclical encoding for time features (preserves continuity)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
        
        # Fill any new NaN values introduced with 0
        # More sophisticated fill method for microstructure features
        for col in df.columns:
            if df[col].isnull().any():
                # Use forward fill first
                df[col] = df[col].fillna(method='ffill')
                # Then any remaining NaNs with zeros
                df[col] = df[col].fillna(0)
        
        logger.info(f"Added derived features and microstructure metrics. New shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding derived features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return original df if feature engineering fails
        return df

def select_best_features(df, target_col='target', n_features=30, method='mutual_info'):
    """
    Select the most predictive features using various statistical methods
    
    Parameters:
    df (pd.DataFrame): Input dataframe with features and target
    target_col (str): Target column name
    n_features (int): Number of features to select
    method (str): Method for feature selection:
                 - 'mutual_info': Mutual information (non-linear relationships)
                 - 'f_regression': F-statistic (linear relationships)
                 - 'correlation': Spearman rank correlation (monotonic relationships)
                 - 'combined': Combine multiple methods with rank aggregation
    
    Returns:
    list: List of selected feature names
    dict: Feature importance scores
    """
    logger.info(f"Selecting top {n_features} features using {method} method")
    
    # Prepare data
    y = df[target_col].values
    # Exclude target and non-numeric columns
    X_cols = [col for col in df.columns if col != target_col and df[col].dtype.kind in 'bifc']
    X = df[X_cols].values
    
    # Handle missing or infinite values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Dictionary to store feature scores from different methods
    feature_scores = {}
    
    if method in ['mutual_info', 'combined']:
        # Mutual information (captures non-linear relationships)
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            # Normalize to 0-1
            mi_scores = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
            feature_scores['mutual_info'] = dict(zip(X_cols, mi_scores))
            logger.info("Calculated mutual information scores")
        except Exception as e:
            logger.warning(f"Error calculating mutual information: {e}")
            feature_scores['mutual_info'] = dict(zip(X_cols, np.zeros(len(X_cols))))
    
    if method in ['f_regression', 'combined']:
        # F-statistic (captures linear relationships)
        try:
            f_scores, _ = f_regression(X, y)
            # Handle potential NaN values
            f_scores = np.nan_to_num(f_scores)
            # Normalize to 0-1
            f_scores = f_scores / np.max(f_scores) if np.max(f_scores) > 0 else f_scores
            feature_scores['f_regression'] = dict(zip(X_cols, f_scores))
            logger.info("Calculated F-regression scores")
        except Exception as e:
            logger.warning(f"Error calculating F-regression: {e}")
            feature_scores['f_regression'] = dict(zip(X_cols, np.zeros(len(X_cols))))
    
    if method in ['correlation', 'combined']:
        # Spearman rank correlation (captures monotonic relationships)
        try:
            correlation_scores = []
            for i in range(X.shape[1]):
                corr, _ = spearmanr(X[:, i], y)
                # Handle NaN correlation values
                if np.isnan(corr):
                    corr = 0
                correlation_scores.append(abs(corr))  # Use absolute correlation
            
            # Normalize to 0-1
            correlation_scores = np.array(correlation_scores)
            correlation_scores = correlation_scores / np.max(correlation_scores) if np.max(correlation_scores) > 0 else correlation_scores
            feature_scores['correlation'] = dict(zip(X_cols, correlation_scores))
            logger.info("Calculated Spearman correlation scores")
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            feature_scores['correlation'] = dict(zip(X_cols, np.zeros(len(X_cols))))
    
    # Combine scores based on selected method
    if method == 'combined':
        # Simple rank aggregation: average of normalized scores from multiple methods
        combined_scores = {}
        for feature in X_cols:
            score = np.mean([
                feature_scores['mutual_info'].get(feature, 0),
                feature_scores['f_regression'].get(feature, 0),
                feature_scores['correlation'].get(feature, 0)
            ])
            combined_scores[feature] = score
            
        # Sort by combined score
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    else:
        # Use the single selected method
        selected_scores = feature_scores[method]
        sorted_features = sorted(selected_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top n features
    top_features = [feature for feature, score in sorted_features[:n_features]]
    logger.info(f"Selected {len(top_features)} top features")
    
    # Create a dictionary with all scores for reporting
    all_scores = {feature: {} for feature in X_cols}
    for method_name, scores in feature_scores.items():
        for feature, score in scores.items():
            all_scores[feature][method_name] = score
    
    # Add rank information
    for method_name in feature_scores.keys():
        method_ranks = {feature: rank for rank, (feature, _) in 
                        enumerate(sorted(feature_scores[method_name].items(), 
                                        key=lambda x: x[1], reverse=True), 1)}
        for feature in X_cols:
            all_scores[feature][f"{method_name}_rank"] = method_ranks.get(feature, len(X_cols))
    
    return top_features, all_scores

def prepare_data_for_tft(df, context_length=30, prediction_length=1, scalers=None, 
                         feature_selection=False, n_features=30, selection_method='combined'):
    """
    Prepare data for the improved TFT model
    
    Parameters:
    df (pd.DataFrame): Input dataframe with all features
    context_length (int): Number of time steps to use as context
    prediction_length (int): Number of time steps to predict
    scalers (dict): Optional pre-fitted scalers for validation/test data
    feature_selection (bool): Whether to apply feature selection
    n_features (int): Number of features to select if feature_selection is True
    selection_method (str): Method for feature selection
    
    Returns:
    tuple: Processed dataframe, feature definitions, fitted scalers
    """
    # Define feature categories
    static_categorical_cols = []  # No static categorical features in our data
    static_real_cols = []  # No static real features in our data
    
    # Potentially convert hour/minute to categorical if they exist
    temporal_categorical_cols = []
    if 'hour' in df.columns:
        temporal_categorical_cols.append('hour')
    if 'minute' in df.columns:
        # Convert to 5-minute buckets to reduce cardinality
        if 'minute' in df.columns:
            df['minute_bucket'] = (df['minute'] // 5).astype(int)
            temporal_categorical_cols.append('minute_bucket')
    
    # Define target column
    target_col = 'target'
    
    # Apply feature selection if requested
    selected_features = None
    feature_scores = None
    
    if feature_selection:
        # Get all numeric columns except categorical and target
        all_potential_features = [col for col in df.columns 
                                if col not in static_categorical_cols + static_real_cols + 
                                temporal_categorical_cols + [target_col]]
        
        logger.info(f"Running feature selection on {len(all_potential_features)} potential features")
        selected_features, feature_scores = select_best_features(
            df, target_col=target_col, n_features=n_features, method=selection_method
        )
        
        # Use only selected features
        temporal_real_cols = selected_features
        logger.info(f"Using {len(temporal_real_cols)} selected features")
    else:
        # Time-varying real features (all numeric features except target)
        temporal_real_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'bid_vol', 'ask_vol', 'imbalance',
            'spread_mean', 'spread_pct_mean',
            'new_bid_orders', 'new_ask_orders',
            'canceled_bid_orders', 'canceled_ask_orders',
            'executed_bid_orders', 'executed_ask_orders',
            'buy_sell_imbalance',
            'returns_10sec', 'returns_30sec', 'returns_1min',
            'volatility_1min',
            # Derived features
            'sma_fast', 'sma_slow', 'trend_indicator',
            'high_volatility', 'imbalance_change', 'cum_imbalance_1min',
            'rsi_14'
        ]
        
        # Add the new microstructure features if they exist
        new_microstructure_features = [
            'norm_imbalance', 'imbalance_acceleration', 'cum_imbalance_3min',
            'new_orders_ratio', 'log_order_ratio', 'bid_cancel_ratio',
            'ask_cancel_ratio', 'relative_cancel_ratio', 'log_cancel_ratio',
            'inv_liquidity', 'spread_change', 'spread_z_score',
            'executed_orders_imbalance', 'normalized_exec_imbalance',
            'returns_per_volume', 'market_impact', 'vol_imbalance_interaction',
            'sign_consistency', 'aligned_flow', 'aligned_flow_cumul',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
        ]
        
        for feature in new_microstructure_features:
            if feature in df.columns:
                temporal_real_cols.append(feature)
    
    # Only keep columns that actually exist in the data
    temporal_real_cols = [col for col in temporal_real_cols if col in df.columns]
    
    # Create a copy of the dataframe with only the needed columns
    all_cols = static_categorical_cols + static_real_cols + temporal_categorical_cols + temporal_real_cols + [target_col]
    df_subset = df[all_cols].copy()
    
    # Normalize time-varying real features
    if scalers is None:
        # Fit new scalers
        scalers = {}
        for col in temporal_real_cols:
            # Check for invalid values
            if df_subset[col].isnull().any() or np.isinf(df_subset[col]).any():
                logger.warning(f"Column {col} contains invalid values. Replacing with 0.")
                df_subset[col] = df_subset[col].replace([np.inf, -np.inf, np.nan], 0)
            
            # Create and fit scaler
            scaler = StandardScaler()
            df_subset[col] = scaler.fit_transform(df_subset[[col]])
            scalers[col] = scaler
    else:
        # Use pre-fitted scalers
        for col in temporal_real_cols:
            if col in scalers:
                # Check for invalid values
                if df_subset[col].isnull().any() or np.isinf(df_subset[col]).any():
                    df_subset[col] = df_subset[col].replace([np.inf, -np.inf, np.nan], 0)
                
                # Transform using existing scaler
                df_subset[col] = scalers[col].transform(df_subset[[col]])
            else:
                logger.warning(f"No scaler found for column {col}, fitting a new one")
                scaler = StandardScaler()
                df_subset[col] = scaler.fit_transform(df_subset[[col]])
                scalers[col] = scaler
    
    # Convert categorical columns to integers
    for col in temporal_categorical_cols:
        if col in df_subset.columns:
            df_subset[col] = df_subset[col].astype(int)
    
    # Feature definitions for model configuration
    feature_definitions = {
        'static_categorical_cols': static_categorical_cols,
        'static_real_cols': static_real_cols,
        'temporal_categorical_cols': temporal_categorical_cols,
        'temporal_real_cols': temporal_real_cols,
        'target_col': target_col,
        'feature_scores': feature_scores  # Include feature importance scores if available
    }
    
    return df_subset, feature_definitions, scalers

def train_improved_tft(
    df,
    feature_definitions,
    context_length=30,
    prediction_length=1,
    hidden_size=128,
    num_attention_heads=4,
    dropout=0.1,
    learning_rate=1e-4,
    batch_size=64,
    max_epochs=100,
    patience=15,
    use_gpu=True,
    save_model_summary=True
):
    """
    Train an improved TFT model
    
    Parameters:
    df (pd.DataFrame): Processed dataframe
    feature_definitions (dict): Feature category definitions
    context_length (int): Number of time steps to use as context
    prediction_length (int): Number of time steps to predict
    hidden_size (int): Hidden dimension size
    num_attention_heads (int): Number of attention heads
    dropout (float): Dropout rate
    learning_rate (float): Initial learning rate
    batch_size (int): Batch size
    max_epochs (int): Maximum number of training epochs
    patience (int): Patience for early stopping
    use_gpu (bool): Whether to use GPU for training
    save_model_summary (bool): Whether to save model summary and feature importance
    
    Returns:
    TemporalFusionTransformer: Trained model
    dict: Training history and metadata
    """
    # Create data loaders
    train_loader, val_loader, test_loader = create_improved_data_loaders(
        data=df,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        static_categorical_cols=feature_definitions['static_categorical_cols'],
        static_real_cols=feature_definitions['static_real_cols'],
        temporal_categorical_cols=feature_definitions['temporal_categorical_cols'],
        temporal_real_cols=feature_definitions['temporal_real_cols'],
        target_col=feature_definitions['target_col'],
        num_workers=0  # Increase for multi-core systems
    )
    
    # Calculate embedding sizes for categorical variables
    # Rule of thumb: min(50, (cardinality+1)//2)
    time_varying_embedding_sizes = []
    for col in feature_definitions['temporal_categorical_cols']:
        cardinality = int(df[col].max()) + 1
        emb_size = min(50, (cardinality + 1) // 2)
        time_varying_embedding_sizes.append((cardinality, emb_size))
    
    logger.info(f"Time-varying categorical embeddings: {time_varying_embedding_sizes}")
    
    # Initialize model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = TemporalFusionTransformer(
        static_variables=[],  # No static variables in our data
        time_varying_categorical_variables=[s[0] for s in time_varying_embedding_sizes] if time_varying_embedding_sizes else [],
        time_varying_real_variables=len(feature_definitions['temporal_real_cols']),
        static_embedding_sizes=[],  # No static embeddings
        time_varying_embedding_sizes=time_varying_embedding_sizes,
        hidden_size=hidden_size,
        lstm_layers=2,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        context_length=context_length,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.5, 0.9],
        loss_fn="mse",  # Use MSE for now, can switch to quantile for probabilistic forecasts
        bias_correction=True
    )
    
    # Create model checkpoint callback
    checkpoint_dir = f"checkpoints/improved_tft_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='tft-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up TensorBoard logger
    logger_dir = f"lightning_logs/improved_tft_{timestamp}"
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    # Try to set up TensorBoard logger
    try:
        tb_logger = TensorBoardLogger(save_dir=logger_dir)
        logger_to_use = tb_logger
    except (ModuleNotFoundError, ImportError):
        logger.warning("TensorBoard not available, using CSVLogger instead")
        from pytorch_lightning.loggers import CSVLogger
        csv_logger = CSVLogger(save_dir=logger_dir)
        logger_to_use = csv_logger
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger_to_use,
        gradient_clip_val=1.0,  # Prevent gradient explosion
        log_every_n_steps=50
    )
    
    # Train model
    logger.info(f"Starting training with {len(train_loader.dataset)} samples")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Test model
    logger.info(f"Evaluating model on test set with {len(test_loader.dataset)} samples")
    test_results = trainer.test(model, dataloaders=test_loader)[0]
    
    # Save feature importance information if available
    if save_model_summary and 'feature_scores' in feature_definitions and feature_definitions['feature_scores']:
        feature_import_path = os.path.join(checkpoint_dir, 'feature_importance.json')
        with open(feature_import_path, 'w') as f:
            # Convert any numpy values to Python types for JSON serialization
            feature_scores = feature_definitions['feature_scores']
            serializable_scores = {}
            for feature, scores in feature_scores.items():
                serializable_scores[feature] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in scores.items()
                }
            json.dump(serializable_scores, f, indent=2)
        logger.info(f"Feature importance saved to {feature_import_path}")
    
    # Save model metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'improved_tft',
        'hyperparameters': {
            'context_length': context_length,
            'prediction_length': prediction_length,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_epochs': max_epochs
        },
        'feature_definitions': {
            'static_categorical_cols': feature_definitions['static_categorical_cols'],
            'static_real_cols': feature_definitions['static_real_cols'],
            'temporal_categorical_cols': feature_definitions['temporal_categorical_cols'],
            'temporal_real_cols': feature_definitions['temporal_real_cols'],
            'target_col': feature_definitions['target_col']
        },
        'time_varying_embedding_sizes': time_varying_embedding_sizes,
        'best_val_loss': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        'best_epoch': checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0] if checkpoint_callback.best_model_path else None,
        'test_results': test_results
    }
    
    # Save metadata
    metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Load best model for saving
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # Save model to models directory
        os.makedirs('models', exist_ok=True)
        model_path = f"models/improved_tft_{timestamp}.pt"
        
        # Save entire model, config, and metadata
        model_package = {
            'model_state_dict': best_model.state_dict(),
            'config': best_model.hparams,
            'metadata': metadata,
            'feature_definitions': {
                'static_categorical_cols': feature_definitions['static_categorical_cols'],
                'static_real_cols': feature_definitions['static_real_cols'],
                'temporal_categorical_cols': feature_definitions['temporal_categorical_cols'],
                'temporal_real_cols': feature_definitions['temporal_real_cols'],
                'target_col': feature_definitions['target_col']
            }
        }
        torch.save(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return best_model, metadata
    else:
        logger.warning("No checkpoint found, returning last model state")
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        model_path = f"models/improved_tft_{timestamp}.pt"
        
        # Save entire model, config, and metadata
        model_package = {
            'model_state_dict': model.state_dict(),
            'config': model.hparams,
            'metadata': metadata,
            'feature_definitions': {
                'static_categorical_cols': feature_definitions['static_categorical_cols'],
                'static_real_cols': feature_definitions['static_real_cols'],
                'temporal_categorical_cols': feature_definitions['temporal_categorical_cols'],
                'temporal_real_cols': feature_definitions['temporal_real_cols'],
                'target_col': feature_definitions['target_col']
            }
        }
        torch.save(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, metadata

def save_model(model, config, feature_definitions, test_metrics, output_path=None):
    """
    Save the model with metadata
    
    Args:
        model: Trained model
        config: Model configuration
        feature_definitions: Feature definitions
        test_metrics: Metrics from test set
        output_path: Output path (if None, will generate one based on timestamp)
    
    Returns:
        Path where the model was saved
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"models/improved_tft_{timestamp}.pt"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_definitions': feature_definitions,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_metrics': test_metrics,
            'pytorch_version': torch.__version__
        }
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Train improved TFT model')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use')
    parser.add_argument('--missing', type=str, default='interpolate', choices=['fill_zero', 'fill_mean', 'fill_forward', 'interpolate'], help='Method for handling missing values')
    parser.add_argument('--context', type=int, default=10, help='Context length (number of time steps)')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden size')
    parser.add_argument('--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training even if GPU is available')
    parser.add_argument('--feature-selection', action='store_true', help='Use feature selection')
    parser.add_argument('--n-features', type=int, default=20, help='Number of features to select if using feature selection')
    parser.add_argument('--selection-method', type=str, default='combined', 
                        choices=['mutual_info', 'f_regression', 'correlation', 'combined'],
                        help='Feature selection method')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.days} days of data with {args.missing} missing value handling...")
    data = load_data(days=args.days, handle_missing=args.missing)
    
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    logger.info(f"Data loaded successfully with shape {data.shape}")
    
    # Prepare data for TFT
    logger.info("Preparing data for TFT...")
    prepared_data, feature_definitions, _ = prepare_data_for_tft(
        data,
        context_length=args.context,
        prediction_length=1,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
        selection_method=args.selection_method
    )
    
    # Log feature selection results if used
    if args.feature_selection and 'feature_scores' in feature_definitions:
        # Find top 5 features by combined score
        top_features = sorted([
            (feature, scores.get('combined', 0)) 
            for feature, scores in feature_definitions['feature_scores'].items()
        ], key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"Top 5 features selected: {top_features}")
    
    logger.info(f"Data prepared with {len(prepared_data)} samples and {len(feature_definitions['temporal_real_cols'])} temporal real features")
    
    # Train model
    logger.info("Starting model training...")
    model, metadata = train_improved_tft(
        df=prepared_data,
        feature_definitions=feature_definitions,
        context_length=args.context,
        prediction_length=1,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch,
        max_epochs=args.epochs,
        patience=args.patience,
        use_gpu=not args.cpu,
        save_model_summary=True
    )
    
    logger.info("Training completed!")
    logger.info(f"Test metrics: {metadata['test_results']}")
    
    # Save model
    save_model(model, model.hparams, feature_definitions, metadata['test_results'])
    
if __name__ == "__main__":
    main() 