from contextlib import redirect_stderr, redirect_stdout
import io
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from datetime import datetime
import logging
from tqdm import tqdm

class TradingMLClassifier:
    """
    Machine Learning classifier for trading decisions based on historical data and predictions.
    This class handles dataset creation, training, evaluation, and prediction for trading signals.
    """
    def __init__(self, preprocessed_df=None, original_df=None):
        """
        Initialize the classifier with preprocessed and original data.
        Args:
            preprocessed_df: DataFrame with preprocessed features
            original_df: DataFrame with original price data (used for labeling)
        """
        self.preprocessed_df = preprocessed_df
        self.original_df = original_df
        self.model = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler for better handling of outliers
        self.lookback_period = 100  # Number of previous timestamps to use
        self.prediction_horizon = 12  # Number of predicted values to use

    def create_dataset(self, stop_loss_pct=0.02, take_profit_pct=0.02, validation_size=0.2, test_size=0.2, random_state=42,
                    save_dataset=True, dataset_path='ressource/dataset/ml_dataset.npz', load_if_exists=True, 
                    prediction_model_path='ressource/models/prediction_model.pkl'):
        """
        Create labeled dataset based on backtest strategy.
        Args:
            stop_loss_pct: Stop loss percentage (default 2%)
            take_profit_pct: Take profit percentage (default 2%)
            validation_size: Size of validation set
            test_size: Size of test set
            random_state: Random seed for reproducibility
            save_dataset: Whether to save the created dataset
            dataset_path: Path to save/load the dataset
            load_if_exists: Whether to load existing dataset if available
            prediction_model_path: Path to the NHITS prediction model
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Train and test datasets
        """
        # Check if we should load an existing dataset
        if load_if_exists and os.path.exists(dataset_path):
            print(f"Loading existing dataset from {dataset_path}...")
            try:
                loaded_data = np.load(dataset_path, allow_pickle=True)
                # Load all arrays
                X_train_scaled = loaded_data['X_train']
                X_val_scaled = loaded_data['X_val'] 
                X_test_scaled = loaded_data['X_test']
                y_train = loaded_data['y_train']
                y_val = loaded_data['y_val']
                y_test = loaded_data['y_test']
                
                # Load feature scaler if available
                if 'feature_scaler.pkl' in loaded_data.files:
                    self.feature_scaler = loaded_data['feature_scaler.pkl'].item()
                
                # Set class attributes
                self.X_train = X_train_scaled
                self.X_val = X_val_scaled
                self.X_test = X_test_scaled
                self.y_train = y_train
                self.y_val = y_val
                self.y_test = y_test
                
                print(f"Dataset loaded successfully!")
                print(f"Train set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}, Test set: {X_test_scaled.shape}")
                print(f"Train labels - Buy: {sum(y_train)}, Hold: {len(y_train)-sum(y_train)}")
                print(f"Test labels - Buy: {sum(y_test)}, Hold: {len(y_test)-sum(y_test)}")
                
                return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
            except Exception as e:
                print(f"Error loading dataset: {e}. Creating new dataset...")
        
        # If we get here, we need to create a new dataset
        if self.preprocessed_df is None or self.original_df is None:
            raise ValueError("Both preprocessed_df and original_df must be provided to create dataset")
        
        print("Creating new dataset...")
        
        # Load NHITS prediction model
        print(f"Loading NHITS prediction model from {prediction_model_path}...")
        try:
            prediction_model = joblib.load(prediction_model_path)
            print("NHITS prediction model loaded successfully")
        except Exception as e:
            print(f"Error loading prediction model: {e}. Continuing without predictions...")
            prediction_model = None
        
        # Ensure timestamp columns are datetime
        self.preprocessed_df['ds'] = pd.to_datetime(self.preprocessed_df['ds'])
        self.original_df['timestamp'] = pd.to_datetime(self.original_df['timestamp'])
        
        # Create features and labels
        features = []
        labels = []
        timestamps = []
        
        # Define eligible timestamps (those with enough history)
        eligible_timestamps = self.preprocessed_df['ds'].iloc[self.lookback_period:]
        print(f"Processing {len(eligible_timestamps)} timestamps...")
        
        for idx, current_timestamp in enumerate(tqdm(eligible_timestamps, desc="Processing timestamps", unit="timestamp")):
            try:
                # Get previous timestamps data
                prev_data = self.preprocessed_df[self.preprocessed_df['ds'] < current_timestamp].tail(self.lookback_period)
                
                # Skip if we don't have enough history
                if len(prev_data) < self.lookback_period:
                    continue
                    
                # Get price at current timestamp from original data
                current_price_data = self.original_df[self.original_df['timestamp'] >= current_timestamp].iloc[0:1]
                if len(current_price_data) == 0:
                    continue
                current_price = current_price_data['close'].values[0]
                
                # Get future price data for labeling
                future_prices = self.original_df[self.original_df['timestamp'] > current_timestamp]['close'].values
                if len(future_prices) < 2:
                    continue
                    
                # Define stop loss and take profit levels
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
                
                # Determine if this would be a good trade (1 = buy, 0 = hold)
                label = 0  # Default to hold
                for future_price in future_prices:
                    if future_price >= take_profit:
                        label = 1  # Buy signal (would hit take profit)
                        break
                    elif future_price <= stop_loss:
                        label = 0  # Hold signal (would hit stop loss)
                        break
                        
                # Extract features (flattened previous data)
                expected_features = ['volume', 'trade_count', 'y', 'SMA_20_normalized',
                                'SMA_50_normalized', 'EMA_20_normalized',
                                'BBL_20_2.0_normalized', 'BBM_20_2.0_normalized',
                                'BBU_20_2.0_normalized', 'RSI_normalized',
                                'MACD_normalized', 'MACD_signal_normalized',
                                'MACD_hist_normalized']
                
                # Ensure all expected features exist in the DataFrame
                missing_features = [col for col in expected_features if col not in prev_data.columns]
                if missing_features:
                    print(f"Warning: Missing features in training data: {missing_features}")
                    continue
                
                # Check for NaN or infinite values and skip if found
                if prev_data[expected_features].isnull().values.any() or np.isinf(prev_data[expected_features].values).any():
                    continue
                
                # Extract features from previous timestamps in a consistent order
                feature_vector = prev_data[expected_features].values.flatten()
                
                # Generate prediction using NHITS model if available
                if prediction_model is not None:
                    try:
                        with io.StringIO() as f, redirect_stdout(f), redirect_stderr(f):
                            # Disable ALL loggers
                            logging.disable(logging.CRITICAL)
                            
                            # Make the prediction
                            prediction = prediction_model.predict(prev_data)   

                            # Re-enable logging after prediction
                            logging.disable(logging.NOTSET)
                        
                        # Extract prediction features
                        pred_values = prediction['NHITS'].values
                        
                        # Current price normalized
                        current_price_normalized = prev_data['y'].iloc[-1]
                        
                        # Calculate prediction features
                        pred_direction = np.array([1 if v > current_price_normalized else 0 for v in pred_values])
                        pred_pct_change = (pred_values - current_price_normalized) / current_price_normalized
                        
                        # Check for NaN or infinite values in predictions
                        if np.isnan(pred_values).any() or np.isinf(pred_values).any() or \
                           np.isnan(pred_pct_change).any() or np.isinf(pred_pct_change).any():
                            continue
                        
                        # Create prediction features array
                        prediction_features = np.concatenate([
                            pred_values,                  # Raw predicted values
                            pred_direction,               # Direction (up/down)
                            pred_pct_change,              # Percentage change
                            [sum(pred_direction)/len(pred_direction)],  # Proportion of up predictions
                            [np.mean(pred_pct_change)],   # Average predicted change
                            [np.std(pred_pct_change)]     # Volatility of predictions
                        ])
                        
                        # Add prediction features to feature vector
                        feature_vector = np.concatenate([feature_vector, prediction_features])
                        
                    except Exception as e:
                        # Continue without prediction features
                        pass
                
                # Check for NaN or infinite values in the final feature vector
                if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
                    continue
                
                # Add to our dataset
                features.append(feature_vector)
                labels.append(label)
                timestamps.append(current_timestamp)
                
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Convert to numpy arrays
        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        
        print(f"Dataset created with {len(features)} samples")
        print(f"Class distribution - Buy: {sum(y)}, Hold: {len(y)-sum(y)}")
        
        # Create train/validation/test split
        X_temp, X_test, y_temp, y_test, temp_timestamps, test_timestamps = train_test_split(
            X, y, timestamps, test_size=test_size, random_state=random_state, stratify=y)
        
        X_train, X_val, y_train, y_val, train_timestamps, val_timestamps = train_test_split(
            X_temp, y_temp, temp_timestamps, test_size=validation_size/(1-test_size),
            random_state=random_state, stratify=y_temp)
        
        # Scale features
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Store the datasets
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_timestamps = train_timestamps
        self.val_timestamps = val_timestamps
        self.test_timestamps = test_timestamps
        
        # Save the dataset if requested
        if save_dataset:
            print(f"Saving dataset to {dataset_path}...")
            try:
                # Save to npz file
                np.savez_compressed(
                    dataset_path,
                    X_train=X_train_scaled,
                    X_val=X_val_scaled,
                    X_test=X_test_scaled,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    train_timestamps=train_timestamps,
                    val_timestamps=val_timestamps,
                    test_timestamps=test_timestamps
                )
                
                # Save feature scaler separately
                joblib.dump(self.feature_scaler, os.path.splitext(dataset_path)[0] + '_scaler.pkl')
                
                print(f"Dataset saved successfully to {dataset_path}")
            except Exception as e:
                print(f"Error saving dataset: {e}")
        
        print(f"Train set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Train labels - Buy: {sum(y_train)}, Hold: {len(y_train)-sum(y_train)}")
        print(f"Test labels - Buy: {sum(y_test)}, Hold: {len(y_test)-sum(y_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def train_models(self, class_weight='balanced'):
        """
        Train multiple ML models and select the best one.
        Args:
            class_weight: Weight adjustment for imbalanced classes
        Returns:
            best_model: The best performing model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset must be created before training models")
            
        print("Training models...")
        
        # Define models to train
        models = {
            'RandomForest': RandomForestClassifier(class_weight=class_weight, n_jobs=-1, random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'LogisticRegression': LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        }
        
        # Define parameter grids for hyperparameter tuning
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga']
            }
        }
        
        best_score = 0
        self.best_model = None
        best_model_name = None
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1 if model_name not in ['XGBoost'] else 1  # Some models don't support parallelization
            )
            
            # Fit the model
            grid_search.fit(self.X_train, self.y_train)
            
            # Evaluate on validation set
            y_pred = grid_search.predict(self.X_val)
            f1 = f1_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred)
            recall = recall_score(self.y_val, y_pred)
            accuracy = accuracy_score(self.y_val, y_pred)
            
            results[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
            
            print(f"{model_name} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                self.best_model = grid_search.best_estimator_
                best_model_name = model_name
        
        print(f"\nBest model: {best_model_name} with F1 score: {best_score:.4f}")
        
        # Evaluate best model on test set
        y_pred = self.best_model.predict(self.X_test)
        test_f1 = f1_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Test metrics - F1: {test_f1:.4f}, Precision: {test_precision:.4f}, "
            f"Recall: {test_recall:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        self.results = results
        return self.best_model

    def evaluate_model(self, model=None, X=None, y=None, threshold=0.5):
        """
        Evaluate a model's performance with detailed metrics.
        Args:
            model: Model to evaluate (uses best_model if None)
            X: Features to use for evaluation (uses X_test if None)
            y: Labels to use for evaluation (uses y_test if None)
            threshold: Probability threshold for classification
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            model = self.best_model
        if model is None:
            raise ValueError("No model available for evaluation")
            
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if X is None or y is None:
            raise ValueError("No data available for evaluation")
            
        # Get predictions
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Hold', 'Buy']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_prob': y_prob,
            'y_pred': y_pred
        }

    def backtest_strategy(self, initial_balance=10000, position_size_pct=0.1):
        """
        Backtest the trading strategy on test data.
        Args:
            initial_balance: Initial account balance
            position_size_pct: Percentage of balance to use per trade
        Returns:
            DataFrame of trades and performance metrics
        """
        if self.best_model is None:
            raise ValueError("No model available for backtesting")
        if self.X_test is None or self.y_test is None or self.test_timestamps is None:
            raise ValueError("No test data available for backtesting")
            
        # Make predictions on test set
        y_pred = self.best_model.predict(self.X_test)
        y_prob = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Create backtest DataFrame
        backtest_df = pd.DataFrame({
            'timestamp': self.test_timestamps,
            'actual_signal': self.y_test,
            'predicted_signal': y_pred,
            'predicted_probability': y_prob
        })
        
        # Sort by timestamp
        backtest_df = backtest_df.sort_values('timestamp')
        
        # Add price data
        backtest_df['entry_price'] = 0.0
        backtest_df['exit_price'] = 0.0
        backtest_df['stop_loss'] = 0.0
        backtest_df['take_profit'] = 0.0
        backtest_df['result'] = 'None'  # Win, Loss, None
        backtest_df['profit_pct'] = 0.0
        
        # Simulation variables
        balance = initial_balance
        position_size = balance * position_size_pct
        in_position = False
        entry_price = 0
        entry_index = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        
        # Ensure timestamp column in original_df is datetime
        self.original_df['timestamp'] = pd.to_datetime(self.original_df['timestamp'])
        
        # Loop through timestamps
        for i, row in backtest_df.iterrows():
            current_timestamp = pd.to_datetime(row['timestamp'])  # Ensure timestamp is datetime
            signal = row['predicted_signal']
            
            # Get price at current timestamp from original data
            price_data = self.original_df[self.original_df['timestamp'] >= current_timestamp].iloc[0:1]
            if len(price_data) == 0:
                continue
            current_price = price_data['close'].values[0]
            
            # If no position and signal is buy
            if not in_position and signal == 1:
                in_position = True
                entry_price = current_price
                entry_index = i
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.02  # 2% take profit
                
                backtest_df.at[i, 'entry_price'] = entry_price
                backtest_df.at[i, 'stop_loss'] = stop_loss
                backtest_df.at[i, 'take_profit'] = take_profit
                
                trade = {
                    'entry_time': current_timestamp,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size
                }
            
            # If in position, check for exit conditions
            elif in_position:
                # Check if stop loss or take profit hit
                if current_price <= stop_loss:
                    # Stop loss hit
                    in_position = False
                    exit_price = current_price
                    profit_pct = (exit_price - entry_price) / entry_price
                    profit_amount = position_size * profit_pct
                    balance += profit_amount
                    
                    backtest_df.at[entry_index, 'exit_price'] = exit_price
                    backtest_df.at[entry_index, 'result'] = 'Loss'
                    backtest_df.at[entry_index, 'profit_pct'] = profit_pct
                    
                    trade['exit_time'] = current_timestamp
                    trade['exit_price'] = exit_price
                    trade['profit_pct'] = profit_pct
                    trade['profit_amount'] = profit_amount
                    trade['result'] = 'Loss'
                    trade['balance'] = balance
                    trades.append(trade)
                
                elif current_price >= take_profit:
                    # Take profit hit
                    in_position = False
                    exit_price = current_price
                    profit_pct = (exit_price - entry_price) / entry_price
                    profit_amount = position_size * profit_pct
                    balance += profit_amount
                    
                    backtest_df.at[entry_index, 'exit_price'] = exit_price
                    backtest_df.at[entry_index, 'result'] = 'Win'
                    backtest_df.at[entry_index, 'profit_pct'] = profit_pct
                    
                    trade['exit_time'] = current_timestamp
                    trade['exit_price'] = exit_price
                    trade['profit_pct'] = profit_pct
                    trade['profit_amount'] = profit_amount
                    trade['result'] = 'Win'
                    trade['balance'] = balance
                    trades.append(trade)
        
        # Calculate backtest metrics
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            win_trades = trades_df[trades_df['result'] == 'Win']
            loss_trades = trades_df[trades_df['result'] == 'Loss']
            win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = win_trades['profit_pct'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['profit_pct'].mean() if len(loss_trades) > 0 else 0
            
            print(f"\nBacktest Results:")
            print(f"Initial Balance: ${initial_balance:.2f}")
            print(f"Final Balance: ${balance:.2f}")
            print(f"Total Return: {(balance - initial_balance) / initial_balance * 100:.2f}%")
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate: {win_rate * 100:.2f}%")
            print(f"Average Win: {avg_win * 100:.2f}%")
            print(f"Average Loss: {avg_loss * 100:.2f}%")
            
            # Plot equity curve
            if len(trades_df) > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(trades_df['exit_time'], trades_df['balance'])
                plt.title('Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Balance ($)')
                plt.grid(True)
                plt.tight_layout()
        else:
            print("No trades executed in backtest")
            
        return backtest_df, trades_df if len(trades) > 0 else pd.DataFrame()

    def save_model(self, filepath='ressource/models/trading_ml_model.joblib'):
        """
        Save the trained model to a file.
        Args:
            filepath: Path to save the model
        Returns:
            filepath where model was saved
        """
        if self.best_model is None:
            raise ValueError("No model available to save")
            
        # Create a dictionary with all necessary components
        model_data = {
            'model': self.best_model,
            'feature_scaler': self.feature_scaler,
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'feature_count': self.X_train.shape[1] if self.X_train is not None else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'has_prediction_features': self.X_train.shape[1] > (self.lookback_period * 13) if self.X_train is not None else False
        }
        
        # Save the model
        joblib.dump(model_data, filepath)
        
        # Also save the scaler separately for easier access
        scaler_filepath = filepath.replace('.joblib', '_scaler.joblib')
        joblib.dump(self.feature_scaler, scaler_filepath)
        
        print(f"Model saved to {filepath}")
        print(f"Scaler saved to {scaler_filepath}")
        return filepath

    def load_model(self, filepath='ressource/models/trading_ml_model.joblib'):
        """
        Load a trained model from file.
        Args:
            filepath: Path to the saved model
        Returns:
            The loaded model
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.lookback_period = model_data['lookback_period']
            self.prediction_horizon = model_data['prediction_horizon']
            
            # Load additional metadata if available
            if 'feature_count' in model_data:
                print(f"Model expects {model_data['feature_count']} features")
            if 'has_prediction_features' in model_data:
                print(f"Model uses prediction features: {model_data['has_prediction_features']}")
            
            print(f"Model loaded from {filepath}")
            print(f"Model timestamp: {model_data.get('timestamp', 'Not available')}")
            
            return self.best_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
    def predict(self, df, prediction_model=None, threshold=0.5, assume_prediction_features=False):
        """
        Make predictions using the trained model.
        Args:
            df: DataFrame with latest preprocessed data (at least lookback_period rows)
            prediction_model: Optional NHITS prediction model
            threshold: Probability threshold for buy signals
            assume_prediction_features: If True, assume model was trained with prediction features
        Returns:
            Prediction (1 = buy, 0 = hold) and probability
        """
        if self.best_model is None:
            raise ValueError("No model available for prediction")
            
        if len(df) < self.lookback_period:
            raise ValueError(f"Input DataFrame must have at least {self.lookback_period} rows")
            
        # Get the latest lookback_period rows
        latest_data = df.tail(self.lookback_period)
        
        # Extract only the expected feature columns in the correct order
        expected_features = ['volume', 'trade_count', 'y', 'SMA_20_normalized',
                        'SMA_50_normalized', 'EMA_20_normalized',
                        'BBL_20_2.0_normalized', 'BBM_20_2.0_normalized',
                        'BBU_20_2.0_normalized', 'RSI_normalized',
                        'MACD_normalized', 'MACD_signal_normalized',
                        'MACD_hist_normalized']
        
        # Verify all expected features are present
        missing_features = [col for col in expected_features if col not in latest_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Extract features in the exact order expected by the model
        feature_vector = latest_data[expected_features].values.flatten()
        
        # Add prediction features if the model was trained with them or if assume_prediction_features is True
        should_add_prediction = (
            assume_prediction_features or 
            (prediction_model is not None and hasattr(self, 'X_train') and self.X_train.shape[1] > len(feature_vector))
        )
        
        if should_add_prediction and prediction_model is not None:
            try:
                with io.StringIO() as f, redirect_stdout(f), redirect_stderr(f):
                    # Disable ALL loggers
                    logging.disable(logging.CRITICAL)
                    
                    # Make the prediction
                    prediction = prediction_model.predict(latest_data)   

                    # Re-enable logging after prediction
                    logging.disable(logging.NOTSET)
                
                # Extract prediction features
                pred_values = prediction['NHITS'].values
                
                # Current price normalized
                current_price_normalized = latest_data['y'].iloc[-1]
                
                # Calculate prediction features
                pred_direction = np.array([1 if v > current_price_normalized else 0 for v in pred_values])
                pred_pct_change = (pred_values - current_price_normalized) / current_price_normalized
                
                # Create prediction features array
                prediction_features = np.concatenate([
                    pred_values,                  # Raw predicted values
                    pred_direction,               # Direction (up/down)
                    pred_pct_change,              # Percentage change
                    [sum(pred_direction)/len(pred_direction)],  # Proportion of up predictions
                    [np.mean(pred_pct_change)],   # Average predicted change
                    [np.std(pred_pct_change)]     # Volatility of predictions
                ])
                
                # Add prediction features to feature vector
                feature_vector = np.concatenate([feature_vector, prediction_features])
            except Exception as e:
                print(f"Warning: Could not add prediction features: {e}")
                # If we can't add prediction features but the model expects them,
                # we need to handle this gracefully
                if assume_prediction_features or (hasattr(self, 'X_train') and self.X_train.shape[1] > len(feature_vector)):
                    # Fill with zeros or some reasonable default values
                    expected_length = self.X_train.shape[1] if hasattr(self, 'X_train') else len(feature_vector) + 42
                    padding = np.zeros(expected_length - len(feature_vector))
                    feature_vector = np.concatenate([feature_vector, padding])
        
        # Scale features
        features_scaled = self.feature_scaler.transform(feature_vector.reshape(1, -1))
        
        # Make prediction
        probability = self.best_model.predict_proba(features_scaled)[0, 1]
        prediction = 1 if probability >= threshold else 0
        
        return {
            'prediction': prediction,  # 1 = buy, 0 = hold
            'probability': probability,
            'threshold': threshold,
            'timestamp': latest_data['ds'].iloc[-1]
        }
    
    def feature_importance(self):
        """
        Get feature importances from the best model (if supported).
        Returns:
            DataFrame of feature importances
        """
        if self.best_model is None:
            raise ValueError("No model available")
            
        # Get feature names for technical indicators
        base_feature_columns = ['volume', 'trade_count', 'y', 'SMA_20_normalized',
                            'SMA_50_normalized', 'EMA_20_normalized',
                            'BBL_20_2.0_normalized', 'BBM_20_2.0_normalized',
                            'BBU_20_2.0_normalized', 'RSI_normalized',
                            'MACD_normalized', 'MACD_signal_normalized',
                            'MACD_hist_normalized']
        
        # Expand feature names for each timestamp in lookback period
        feature_names = []
        for i in range(self.lookback_period):
            for col in base_feature_columns:
                feature_names.append(f"t-{self.lookback_period-i}_{col}")
        
        # Add prediction feature names if they exist
        if hasattr(self, 'X_train') and self.X_train.shape[1] > len(feature_names):
            # Calculate number of prediction features
            num_pred_features = self.X_train.shape[1] - len(feature_names)
            
            # Add prediction feature names
            # Basic prediction values
            for i in range(self.prediction_horizon):
                feature_names.append(f"pred_val_t+{i+1}")
            
            # Direction predictions
            for i in range(self.prediction_horizon):
                feature_names.append(f"pred_dir_t+{i+1}")
            
            # Percentage change predictions
            for i in range(self.prediction_horizon):
                feature_names.append(f"pred_pct_change_t+{i+1}")
            
            # Aggregated prediction features
            feature_names.append(f"pred_direction_ratio")
            feature_names.append(f"pred_mean_pct_change")
            feature_names.append(f"pred_volatility")
            
            # If we still have missing feature names, add generic ones
            remaining = self.X_train.shape[1] - len(feature_names)
            if remaining > 0:
                for i in range(remaining):
                    feature_names.append(f"unknown_pred_feature_{i+1}")
        
        # Check if model supports feature_importances_
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print("Model doesn't support feature importance extraction")
            return None
        
        # Create DataFrame
        if len(importances) == len(feature_names):
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance_df['feature'].head(20),
                    feature_importance_df['importance'].head(20))
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            
            return feature_importance_df
        else:
            print(f"Feature importance length mismatch: {len(importances)} vs {len(feature_names)}")
            return None