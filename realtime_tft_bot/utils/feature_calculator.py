import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureCalculator:
    """
    Processes raw market data into features suitable for the TFT model
    """
    
    def __init__(self, config):
        self.config = config
        self.context_length = config['CONTEXT_LENGTH']
        self.feature_history = pd.DataFrame()
        self.latest_features = None
    
    def initialize(self):
        """Initialize feature calculator"""
        logger.info("Feature calculator initialized")
        return True
    
    def update_with_ohlcv(self, ohlcv_df):
        """
        Update features with new OHLCV data
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            True if successful, False otherwise
        """
        if ohlcv_df is None or len(ohlcv_df) == 0:
            logger.warning("Empty OHLCV data provided to feature calculator")
            return False
        
        try:
            # Make a copy to avoid modifying the input
            df = ohlcv_df.copy()
            
            # Ensure required columns are numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.error(f"Missing required column '{col}' in OHLCV data.")
                    return False # Cannot proceed without essential columns
            
            # Drop rows where essential price/volume data is missing after coercion
            df.dropna(subset=['close', 'volume'], inplace=True)
            if df.empty:
                logger.warning("OHLCV data is empty after dropping rows with NaN close/volume.")
                return False

            # Add basic price features
            df['returns'] = df['close'].pct_change()
            
            # Robust log returns calculation
            price_ratio = df['close'] / df['close'].shift(1)
            # Ensure ratio is numeric and handle potential division by zero or non-numeric results
            price_ratio_numeric = pd.to_numeric(price_ratio, errors='coerce')
            # Replace non-positive values with NaN before taking log
            price_ratio_numeric[price_ratio_numeric <= 0] = np.nan 
            df['log_returns'] = np.log(price_ratio_numeric)
            
            df['volatility'] = df['log_returns'].rolling(5).std()
            
            # Add normalized price features
            df['normalized_open'] = df['open'] / df['close'].shift(1) - 1
            df['normalized_high'] = df['high'] / df['close'].shift(1) - 1
            df['normalized_low'] = df['low'] / df['close'].shift(1) - 1
            df['normalized_close'] = df['close'] / df['close'].shift(1) - 1
            
            # Calculate volume features
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_10'] = df['volume'].rolling(10).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma_5']
            
            # Calculate price momentum features
            df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # Calculate moving averages
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_10'] = df['close'].rolling(10).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            
            # Calculate moving average convergence/divergence
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Calculate RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Calculate ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # Calculate Stochastic Oscillator
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Calculate price distance from moving averages
            df['price_to_ma5'] = df['close'] / df['ma_5'] - 1
            df['price_to_ma10'] = df['close'] / df['ma_10'] - 1
            df['price_to_ma20'] = df['close'] / df['ma_20'] - 1
            
            # Add high-low range feature
            df['hl_range'] = (df['high'] - df['low']) / df['close'].shift(1)
            
            # Drop NaN values (resulting from calculations with lookback periods)
            df = df.dropna()
            
            # Store the feature history and the latest features
            self.feature_history = df
            
            # Select most recent row for prediction
            if len(df) > 0:
                self.latest_features = df.iloc[-1].to_dict()
                logger.debug("Updated features with latest OHLCV data")
                return True
            else:
                logger.warning("No valid features calculated after processing OHLCV data")
                return False
                
        except Exception as e:
            logger.error(f"Error calculating features from OHLCV: {type(e).__name__} - {str(e)}")
            return False
    
    def add_order_book_features(self, order_book):
        """
        Add order book features to the latest feature set
        
        Args:
            order_book: Dict with order book data {'bids': {}, 'asks': {}}
            
        Returns:
            True if successful, False otherwise
        """
        if not self.latest_features:
            logger.warning("No base features available. Call update_with_ohlcv first")
            return False
            
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            logger.warning("Invalid order book data provided")
            return False
            
        try:
            # Convert price strings to floats and sort
            bids = [(float(price), data['quantity']) for price, data in order_book['bids'].items()]
            asks = [(float(price), data['quantity']) for price, data in order_book['asks'].items()]
            
            bids = sorted(bids, key=lambda x: -x[0])  # Sort bids in descending order
            asks = sorted(asks, key=lambda x: x[0])   # Sort asks in ascending order
            
            if not bids or not asks:
                logger.warning("Empty order book")
                return False
                
            # Calculate order book features
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            spread_pct = spread / mid_price if mid_price else 0
            
            # Calculate imbalance measures
            bid_volume = sum(quantity for _, quantity in bids[:10])
            ask_volume = sum(quantity for _, quantity in asks[:10])
            total_volume = bid_volume + ask_volume
            volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume else 0
            
            # Calculate weighted prices
            bid_wap = sum(price * quantity for price, quantity in bids[:10]) / bid_volume if bid_volume else 0
            ask_wap = sum(price * quantity for price, quantity in asks[:10]) / ask_volume if ask_volume else 0
            
            # Calculate volume-weighted spread
            vw_spread = ask_wap - bid_wap if bid_wap and ask_wap else 0
            vw_spread_pct = vw_spread / mid_price if mid_price else 0
            
            # Calculate price impact to buy/sell
            def calculate_price_impact(orders, direction, base_amount=10000):
                """Calculate price impact for a market order of base_amount"""
                total_cost = 0
                total_quantity = 0
                
                for price, quantity in orders:
                    if total_quantity >= base_amount:
                        break
                        
                    executable_quantity = min(quantity, base_amount - total_quantity)
                    total_cost += executable_quantity * price
                    total_quantity += executable_quantity
                
                if total_quantity == 0:
                    return 0
                    
                avg_price = total_cost / total_quantity
                if direction == 'buy':
                    return (avg_price / mid_price) - 1 if mid_price else 0
                else:
                    return 1 - (avg_price / mid_price) if mid_price else 0
            
            # Calculate price impact for buy/sell orders
            buy_price_impact = calculate_price_impact(asks, 'buy')
            sell_price_impact = calculate_price_impact(bids, 'sell')
            
            # Add features to the latest feature set
            ob_features = {
                'ob_mid_price': mid_price,
                'ob_spread': spread,
                'ob_spread_pct': spread_pct,
                'ob_volume_imbalance': volume_imbalance,
                'ob_bid_wap': bid_wap,
                'ob_ask_wap': ask_wap,
                'ob_vw_spread': vw_spread,
                'ob_vw_spread_pct': vw_spread_pct,
                'ob_buy_price_impact': buy_price_impact,
                'ob_sell_price_impact': sell_price_impact,
                'ob_bid_volume': bid_volume,
                'ob_ask_volume': ask_volume,
            }
            
            # ENHANCEMENT: Add more depth-level features
            # Calculate price impact at different depth levels
            def calculate_multi_level_impact(orders, direction, levels=[5, 10, 20, 50]):
                impacts = {}
                for level in levels:
                    level_orders = orders[:level] if len(orders) >= level else orders
                    impact = calculate_price_impact(level_orders, direction)
                    impacts[f"{direction}_impact_level_{level}"] = impact
                return impacts
            
            # Add multi-level bid/ask features
            bid_impacts = calculate_multi_level_impact(bids, 'bid')
            ask_impacts = calculate_multi_level_impact(asks, 'ask')
            
            # Calculate order book depth at price levels
            def calculate_depth_at_price_levels(orders, mid_price, direction, levels=[0.1, 0.2, 0.5, 1.0]):
                depth = {}
                for level in levels:
                    # Calculate percentage levels from mid price
                    level_pct = level / 100
                    if direction == 'bids':
                        price_level = mid_price * (1 - level_pct)
                        # Sum quantities for all orders with price >= price_level
                        depth[f"bid_depth_{level}"] = sum(quantity for price, quantity in orders if price >= price_level)
                    else:
                        price_level = mid_price * (1 + level_pct)
                        # Sum quantities for all orders with price <= price_level
                        depth[f"ask_depth_{level}"] = sum(quantity for price, quantity in orders if price <= price_level)
                return depth
            
            # Add depth at different price levels
            bid_depth = calculate_depth_at_price_levels(bids, mid_price, 'bids')
            ask_depth = calculate_depth_at_price_levels(asks, mid_price, 'asks')
            
            # Create combined dict of all new features
            additional_ob_features = {
                **bid_impacts,
                **ask_impacts,
                **bid_depth,
                **ask_depth,
                "bid_ask_ratio": bid_volume / ask_volume if ask_volume else 1.0,
                "bid_ask_size_imbalance": (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
            }
            
            # Add more specific OB features
            book_pressure = 0
            for i, (bid_price, bid_qty) in enumerate(bids[:5]):
                distance_from_mid = (mid_price - bid_price) / mid_price
                book_pressure -= bid_qty * (1 - distance_from_mid)  # bid pressure is negative (downward)
                
            for i, (ask_price, ask_qty) in enumerate(asks[:5]):
                distance_from_mid = (ask_price - mid_price) / mid_price
                book_pressure += ask_qty * (1 - distance_from_mid)  # ask pressure is positive (upward)
                
            additional_ob_features["book_pressure"] = book_pressure
            
            # Calculate liquidity at different price levels
            bid_liquidity = sum(price * quantity for price, quantity in bids[:10])
            ask_liquidity = sum(price * quantity for price, quantity in asks[:10])
            additional_ob_features["bid_liquidity"] = bid_liquidity
            additional_ob_features["ask_liquidity"] = ask_liquidity
            additional_ob_features["liquidity_imbalance"] = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity) if (bid_liquidity + ask_liquidity) > 0 else 0
            
            # Add these additional features to our ob_features
            ob_features.update(additional_ob_features)
            
            # Update latest features with order book features
            self.latest_features.update(ob_features)
            logger.debug("Added order book features to the latest feature set")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating order book features: {type(e).__name__} - {str(e)}")
            return False
    
    def add_microstructure_features(self):
        """Add market microstructure features"""
        if not self.latest_features:
            logger.warning("No base features available for microstructure calculation")
            return False
            
        try:
            # Calculate realized volatility from recent returns
            if 'returns' in self.feature_history.columns:
                recent_returns = self.feature_history['returns'].dropna().tail(20)
                self.latest_features['realized_volatility_20'] = recent_returns.std() * np.sqrt(252)
                
            # Calculate higher moments
            if 'log_returns' in self.feature_history.columns:
                recent_log_returns = self.feature_history['log_returns'].dropna().tail(20)
                self.latest_features['skewness'] = recent_log_returns.skew()
                self.latest_features['kurtosis'] = recent_log_returns.kurtosis()
                
            # Calculate autocorrelation
            if 'returns' in self.feature_history.columns:
                returns_series = self.feature_history['returns'].dropna()
                if len(returns_series) > 5:
                    autocorr_1 = returns_series.autocorr(lag=1)
                    autocorr_2 = returns_series.autocorr(lag=2)
                    self.latest_features['autocorr_1'] = autocorr_1
                    self.latest_features['autocorr_2'] = autocorr_2
            
            logger.debug("Added market microstructure features")
            return True
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {type(e).__name__} - {str(e)}")
            return False
    
    def prepare_model_input(self):
        """
        Prepare the input for the TFT model
        
        Returns:
            Dict with prepared model input or None if not enough data
        """
        if self.feature_history.empty or len(self.feature_history) < self.context_length:
            logger.warning(f"Not enough historical data for context (need {self.context_length} rows)")
            return None
        
        try:
            # Add market microstructure features
            self.add_microstructure_features()
            
            # DEBUG: Log feature status
            self.debug_features()
            
            # Get the last context_length rows
            context_df = self.feature_history.iloc[-self.context_length:].copy()
            
            # Define static features (constant across time steps)
            # In this example, we don't have static features, but in a real implementation
            # this could include things like trading pair properties, market
            # capitalization classification, etc.
            static_features = {}
            
            # Define known input features (features available at prediction time)
            # These are the features used by the model - exactly 64 features
            model_input_features = [
                # Price features
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'log_returns', 'volatility',
                'normalized_open', 'normalized_high', 'normalized_low', 'normalized_close',
                
                # Volume features
                'volume_ma_5', 'volume_ma_10', 'relative_volume',
                
                # Momentum features
                'momentum_1', 'momentum_5', 'momentum_10',
                
                # Moving averages
                'ma_5', 'ma_10', 'ma_20',
                'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
                
                # Technical indicators
                'macd', 'macd_signal', 'macd_histogram',
                'rsi', 'bb_width', 'bb_pct_b',
                'atr', 'atr_pct', 'stoch_k', 'stoch_d',
                'hl_range',
                
                # Order book basic features
                'ob_mid_price', 'ob_spread', 'ob_spread_pct',
                'ob_volume_imbalance', 'ob_bid_wap', 'ob_ask_wap', 
                'ob_vw_spread', 'ob_vw_spread_pct',
                'ob_buy_price_impact', 'ob_sell_price_impact',
                'ob_bid_volume', 'ob_ask_volume',
                
                # Enhanced order book features
                'bid_impact_level_5', 'bid_impact_level_10', 'bid_impact_level_20', 'bid_impact_level_50',
                'ask_impact_level_5', 'ask_impact_level_10', 'ask_impact_level_20', 'ask_impact_level_50',
                'bid_depth_0.1', 'bid_depth_0.2', 'bid_depth_0.5', 'bid_depth_1.0',
                'ask_depth_0.1', 'ask_depth_0.2', 'ask_depth_0.5', 'ask_depth_1.0',
                'bid_ask_ratio', 'bid_ask_size_imbalance'
            ]
            
            # Print full list of available features from latest_features
            logger.info(f"All available latest features: {list(self.latest_features.keys())}")
            
            # Make sure the order book features are available in latest_features
            ob_features = [f for f in self.latest_features.keys() if f.startswith('ob_')]
            impact_features = [f for f in self.latest_features.keys() if 'impact_level' in f]
            depth_features = [f for f in self.latest_features.keys() if 'depth' in f]
            
            # Log extracted feature categories
            logger.info(f"Found {len(ob_features)} OB features: {ob_features}")
            logger.info(f"Found {len(impact_features)} impact features: {impact_features}")
            logger.info(f"Found {len(depth_features)} depth features: {depth_features}")
            
            # Filter features to include only those in our model
            available_features = context_df.columns.tolist()
            filtered_features = [f for f in model_input_features if f in available_features]
            
            # Debug: Log which features are missing from our available features
            missing_from_historical = [f for f in model_input_features if f not in available_features]
            logger.info(f"Features missing from historical data: {missing_from_historical}")
            
            # If we have order book features in the latest_features but not in the historical data,
            # we need to add them to the last row
            ob_features = [f for f in model_input_features if f.startswith('ob_')]
            impact_features = [f for f in model_input_features if 'impact_level' in f]
            depth_features = [f for f in model_input_features if 'depth' in f]
            other_ob_features = [f for f in model_input_features if f in ['bid_ask_ratio', 'bid_ask_size_imbalance']]
            
            # Combine all OB-related features
            all_ob_features = ob_features + impact_features + depth_features + other_ob_features
            
            # Add all OB-related features to context_df
            for feature in all_ob_features:
                if feature in self.latest_features and feature not in available_features:
                    # Add NaN for all rows except the last one
                    context_df[feature] = np.nan
                    # Update the last row with the current value
                    context_df.iloc[-1, context_df.columns.get_loc(feature)] = self.latest_features[feature]
            
            # Filter the feature matrix again after adding order book features
            filtered_features = [f for f in model_input_features if f in context_df.columns]
            
            # Debug: Log which features are still missing after adding OB features
            still_missing = [f for f in model_input_features if f not in context_df.columns]
            logger.info(f"Features still missing after adding OB features: {still_missing}")
            
            # Check if we have enough features
            min_features = self.config.get('MIN_FEATURES', 20)
            if len(filtered_features) < min_features:
                logger.warning(f"Not enough features available. Have {len(filtered_features)}, need at least {min_features}")
                return None
            
            # Select the feature matrix
            x = context_df[filtered_features].values.astype(np.float32)
            
            # Handle NaN values in the feature matrix
            # For the provided context, we'll forward fill and then backfill
            x_df = pd.DataFrame(x)
            x_df = x_df.ffill().bfill()
            x = x_df.values.astype(np.float32)
            
            # Log the exact feature list and count before padding
            logger.info(f"Features before padding/truncation: {len(filtered_features)}")
            for i, feature in enumerate(filtered_features):
                logger.info(f"Feature {i+1}/{len(filtered_features)}: ✓ {feature}")
            
            # Ensure we meet the expected feature dimension by padding if necessary
            expected_feature_count = self.config.get('EXPECTED_FEATURE_COUNT', 64)
            actual_feature_count = x.shape[1]
            
            if actual_feature_count < expected_feature_count:
                logger.warning(f"Feature count mismatch: have {actual_feature_count}, model expects {expected_feature_count}. Padding with zeros.")
                # Create a padded array with zeros
                x_padded = np.zeros((x.shape[0], expected_feature_count), dtype=np.float32)
                # Copy original features to the beginning
                x_padded[:, :actual_feature_count] = x
                # Use the padded array
                x = x_padded
                # Update feature names to reflect padding
                filtered_features += [f'padding_{i}' for i in range(expected_feature_count - actual_feature_count)]
            elif actual_feature_count > expected_feature_count:
                logger.warning(f"Feature count mismatch: have {actual_feature_count}, model expects {expected_feature_count}. Truncating.")
                # Truncate to expected feature count
                x = x[:, :expected_feature_count]
                filtered_features = filtered_features[:expected_feature_count]
            
            # Prepare the model input dictionary
            model_input = {
                'x': x,  # Feature matrix [context_length x num_features]
                'static_features': static_features,  # Static features dict
                'feature_names': filtered_features,  # List of feature names
                'timestamp': datetime.now()  # Current timestamp
            }
            
            logger.info(f"Prepared model input with shape {x.shape} ({len(filtered_features)} features)")
            return model_input
            
        except Exception as e:
            logger.error(f"Error preparing model input: {type(e).__name__} - {str(e)}")
            return None 
    
    def debug_features(self):
        """
        Debug method to identify which features are missing from our expected set
        """
        if not self.latest_features:
            logger.warning("No features available for debugging")
            return False
            
        # Define the expected feature list - should be exactly 64
        expected_features = [
            # Price features
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'volatility',
            'normalized_open', 'normalized_high', 'normalized_low', 'normalized_close',
            
            # Volume features
            'volume_ma_5', 'volume_ma_10', 'relative_volume',
            
            # Momentum features
            'momentum_1', 'momentum_5', 'momentum_10',
            
            # Moving averages
            'ma_5', 'ma_10', 'ma_20',
            'price_to_ma5', 'price_to_ma10', 'price_to_ma20',
            
            # Technical indicators
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'bb_width', 'bb_pct_b',
            'atr', 'atr_pct', 'stoch_k', 'stoch_d',
            'hl_range',
            
            # Order book basic features
            'ob_mid_price', 'ob_spread', 'ob_spread_pct',
            'ob_volume_imbalance', 'ob_bid_wap', 'ob_ask_wap', 
            'ob_vw_spread', 'ob_vw_spread_pct',
            'ob_buy_price_impact', 'ob_sell_price_impact',
            'ob_bid_volume', 'ob_ask_volume',
            
            # Enhanced order book features
            'bid_impact_level_5', 'bid_impact_level_10', 'bid_impact_level_20', 'bid_impact_level_50',
            'ask_impact_level_5', 'ask_impact_level_10', 'ask_impact_level_20', 'ask_impact_level_50',
            'bid_depth_0.1', 'bid_depth_0.2', 'bid_depth_0.5', 'bid_depth_1.0',
            'ask_depth_0.1', 'ask_depth_0.2', 'ask_depth_0.5', 'ask_depth_1.0',
            'bid_ask_ratio', 'bid_ask_size_imbalance'
        ]
        
        # Get current features
        current_features = list(self.latest_features.keys())
        
        # Check which expected features are missing
        missing_features = [f for f in expected_features if f not in current_features]
        
        # Check for unexpected features
        unexpected_features = [f for f in current_features if f not in expected_features]
        
        # Log results
        logger.info(f"Feature count: {len(current_features)} out of {len(expected_features)} expected")
        logger.info(f"Missing features ({len(missing_features)}): {missing_features}")
        
        if unexpected_features:
            logger.info(f"Unexpected features ({len(unexpected_features)}): {unexpected_features}")
        
        # Feature checklist
        for i, feature in enumerate(expected_features):
            status = "✓" if feature in current_features else "✗"
            logger.info(f"Feature {i+1:2d}/{len(expected_features)}: {status} {feature}")
        
        return True 