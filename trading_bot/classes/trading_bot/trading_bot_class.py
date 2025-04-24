from datetime import datetime, timedelta, timezone
import logging
import time
import io
from contextlib import redirect_stderr, redirect_stdout
import os
import sys

# Import existing dependencies
import pandas as pd
import numpy as np
import talib
import torch
import joblib
import matplotlib.pyplot as plt

# Alpaca imports
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

# Add parent directory to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the trading ML classifier
from classes.ml_classifier.trading_ml_classifier import TradingMLClassifier

# Import our new DatabaseHandler
from classes.database.database_handler import DatabaseHandler

class TradingBot:
    """
    Trading bot with ML-based decision making and risk management.
    
    This bot combines time series prediction models with ML classifiers to make trading decisions
    with proper risk management strategies and database persistence.
    """
    
    def __init__(self, 
                ts_prediction_model, 
                ml_model_path,
                symbol, 
                api_key=None, 
                api_secret=None, 
                paper=True, 
                position_size_pct=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.02,
                max_trades=5,
                prediction_interval=15,
                check_interval=1,
                log_level=logging.INFO,
                db_config=None):
        """
        Initialize the trading bot.
        
        Args:
            ts_prediction_model: Time series prediction model (NHITS)
            ml_model_path: Path to trained ML classifier model (required)
            symbol: Trading symbol (e.g., 'BTC/USD')
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: If True, use paper trading
            position_size_pct: Percentage of available cash per trade
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit percentage (e.g., 0.02 for 2%)
            max_trades: Maximum number of active trades allowed
            prediction_interval: Minutes between predictions
            check_interval: Minutes between trade checks
            log_level: Logging level
            db_config: Database configuration dictionary
        """
        # Setup logging
        self.setup_logging(log_level)
        self.logger = logging.getLogger("TradingBot")
        
        # Trading parameters
        self.active_trades = {}
        self.wallet_balance = 0
        self.ts_prediction_model = ts_prediction_model
        self.symbol = symbol
        self.original_df = None
        self.preprocessed_df = None
        self.last_prediction = None
        self.last_action = None
        self.last_prediction_time = None
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades = max_trades
        self.prediction_interval = prediction_interval
        self.check_interval = check_interval
        self.position_size_pct = position_size_pct
        
        # ML model setup - now required
        self.ml_classifier = TradingMLClassifier()
        if ml_model_path:
            self.load_ml_model(ml_model_path)
        else:
            self.logger.error("ML model path is required")
            raise ValueError("ML model path is required")
        
        # Alpaca API setup
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        
        # Initialize trading client if credentials are provided
        if api_key and api_secret:
            self.setup_trading_client()
            self.logger.info(f"Trading client initialized for {symbol} - Paper trading: {paper}")
        else:
            self.trading_client = None
            self.logger.warning("No API credentials provided. Running in simulation mode only.")
        
        # Database setup
        self.db_handler = None
        if db_config:
            self.setup_database(db_config)
        else:
            self.logger.warning("No database configuration provided. Running without database persistence.")
        
        # Performance metrics
        self.trades_history = []
        self.performance_metrics = {
            'win_count': 0,
            'loss_count': 0,
            'total_profit': 0,
            'total_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
        }
                
    def setup_logging(self, log_level):
        """Set up logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )

    def setup_trading_client(self):
        """Initialize the Alpaca trading client"""
        try:
            self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
            self.account = self.trading_client.get_account()
            self.logger.info(f"Trading account ready. Current balance: ${float(self.account.cash)}")
        except Exception as e:
            self.logger.error(f"Failed to initialize trading client: {e}")
            self.trading_client = None
    
    def setup_database(self, db_config):
        """
        Initialize database connection and set up trades table.
        
        Args:
            db_config: Dictionary with database configuration parameters
        """
        try:
            self.db_handler = DatabaseHandler(
                dbname=db_config.get('dbname', 'trading_bot'),
                user=db_config.get('user', 'postgres'),
                password=db_config.get('password', ''),
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', '5432')
            )
            
            # Connect to database
            if self.db_handler.connect():
                # Create trades table if it doesn't exist
                if self.db_handler.create_trades_table():
                    self.logger.info("Database setup completed successfully")
                else:
                    self.logger.warning("Failed to create trades table")
            else:
                self.logger.warning("Failed to connect to database")
                self.db_handler = None
                
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            self.db_handler = None
    
    def load_ml_model(self, model_path):
        """Load an ML classifier model and its scaler"""
        try:
            self.logger.info(f"Loading ML model from {model_path}")
            self.ml_classifier = TradingMLClassifier()
            
            # Load the model
            model_loaded = self.ml_classifier.load_model(model_path)
            
            # Also try to load the scaler separately if it exists
            scaler_path = model_path.replace('.joblib', '_scaler.joblib')
            if os.path.exists(scaler_path):
                try:
                    # Load the scaler
                    self.logger.info(f"Loading separate scaler from {scaler_path}")
                    self.ml_classifier.feature_scaler = joblib.load(scaler_path)
                    self.logger.info("Scaler loaded successfully")
                except Exception as e:
                    self.logger.error(f"Failed to load separate scaler: {e}")
                    # If we can't load the separate scaler, try to extract it from the model
                    if hasattr(self.ml_classifier, 'feature_scaler') and self.ml_classifier.feature_scaler is not None:
                        self.logger.info("Using scaler from model")
                    else:
                        self.logger.error("No valid scaler found")
            
            if model_loaded:
                self.logger.info("ML model loaded successfully")
            else:
                self.logger.error("Failed to load ML model")
                self.ml_classifier = None
                
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            self.ml_classifier = None

    def fetch_historical_data(self, lookback_hours=149):
        """
        Fetch historical data for the specified symbol
        
        Args:
            lookback_hours: Number of hours to look back
            
        Returns:
            DataFrame of historical data
        """
        self.logger.info(f"Fetching {lookback_hours} hours of historical data for {self.symbol}")
        try:
            client = CryptoHistoricalDataClient()
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Hour,
                start=datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
                end=datetime.now(timezone.utc)
            )
            
            bars = client.get_crypto_bars(request_params)
            
            if bars is None or bars.df.empty:
                self.logger.error(f"No historical data returned for {self.symbol}")
                return None
                
            self.original_df = bars.df.reset_index()
            self.logger.info(f"Fetched {len(self.original_df)} bars of historical data")
            
            # Preprocess the data
            self.preprocess_data()
            
            return self.original_df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return None

    def preprocess_data(self):
        """
        Preprocess the historical data for prediction.
        Ensures the same preprocessing steps as in TradingMLClassifier.
        
        Returns:
            DataFrame with preprocessed features
        """
        if self.original_df is None or len(self.original_df) == 0:
            self.logger.error("No data to preprocess")
            return None
            
        self.logger.info("Preprocessing data for prediction")
        df = self.original_df.copy()
        
        try:
            # Calculate technical indicators
            if 'close' in df.columns:
                # Calculate SMA
                df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
                df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
                
                # Calculate EMA
                df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
                
                # Calculate Bollinger Bands
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                df['BBL_20_2.0'] = lower
                df['BBM_20_2.0'] = middle
                df['BBU_20_2.0'] = upper
                
                # Calculate RSI
                df['RSI'] = talib.RSI(df['close'], timeperiod=14)
                
                # Calculate MACD
                macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                df['MACD'] = macd
                df['MACD_signal'] = macd_signal
                df['MACD_hist'] = macd_hist
                
                # Normalize indicators and close price using Min-Max scaling
                columns_to_normalize = ['close', 'SMA_20', 'SMA_50', 'EMA_20', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist']
                
                for col in columns_to_normalize:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[f'{col}_normalized'] = 0
                
                # Keep only the relevant columns (match TradingMLClassifier expectations)
                columns_to_keep = [
                    'volume',
                    'trade_count',
                    'close_normalized',
                    'SMA_20_normalized', 
                    'SMA_50_normalized',
                    'EMA_20_normalized',
                    'BBL_20_2.0_normalized',
                    'BBM_20_2.0_normalized', 
                    'BBU_20_2.0_normalized',
                    'RSI_normalized',
                    'MACD_normalized',
                    'MACD_signal_normalized',
                    'MACD_hist_normalized'
                ]
                
                # Remove NaN values
                processed_df = df[columns_to_keep].dropna()
                
                # Reset index and rename columns
                processed_df = processed_df.reset_index()
                processed_df = processed_df.rename(columns={'timestamp': 'ds', 'close_normalized': 'y', 'symbol': 'unique_id'})
                
                # Ensure 'ds' is datetime
                if 'ds' not in processed_df.columns and 'index' in processed_df.columns:
                    processed_df['ds'] = df['timestamp'].iloc[processed_df.index]
                
                processed_df['ds'] = pd.to_datetime(processed_df['ds'])
                
                # Add unique_id column if it doesn't exist
                if 'unique_id' not in processed_df.columns:
                    processed_df['unique_id'] = self.symbol
                
                self.preprocessed_df = processed_df
                self.logger.info(f"Preprocessing complete. Result shape: {processed_df.shape}")
                return processed_df
            else:
                self.logger.error("Missing 'close' column in data")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in preprocessing data: {e}", exc_info=True)
            return None
        
    def make_prediction(self):
        """
        Make price predictions using the time series model and prepare for ML classification.
        
        Returns:
            bool: True if prediction was successful, False otherwise
        """
        current_time = datetime.now()
        
        # Initialize last_prediction_time if it's None
        if self.last_prediction_time is None:
            self.last_prediction_time = current_time - timedelta(minutes=self.prediction_interval)
            self.logger.info(f"Initialized last_prediction_time to {self.prediction_interval} minutes ago")
        
        # Check if enough time has elapsed since the last prediction
        time_since_last = (current_time - self.last_prediction_time).total_seconds() / 60
        if time_since_last < self.prediction_interval:
            self.logger.debug(f"Skipping prediction, only {time_since_last:.2f} minutes since last prediction")
            return False
        
        # Update last prediction time
        self.last_prediction_time = current_time
        self.logger.info("Making new prediction")
        
        if self.preprocessed_df is None or 'ds' not in self.preprocessed_df.columns:
            self.logger.error("Cannot make prediction: preprocessed data not available")
            return False
        
        try:
            # Get the data for prediction
            df = self.preprocessed_df.copy()
            
            # Ensure 'ds' is datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Make the prediction
            with io.StringIO() as f, redirect_stdout(f), redirect_stderr(f):
                # Temporarily disable loggers
                logging.disable(logging.CRITICAL)
                pred = self.ts_prediction_model.predict(df)
                logging.disable(logging.NOTSET)
            
            # Store raw predictions for ML model
            self.ts_predictions = pred
            
            # Process predictions for ML classifier
            self.last_prediction = df  # We'll use the preprocessed data with the predictions
            self.logger.info(f"Time series prediction made successfully for {self.symbol}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}", exc_info=True)
            return False

    def get_decision(self, threshold=0.5):
        """
        Get a trading decision (buy/hold) using the ML classifier.
        The ML classifier internally handles running the time series prediction model when needed.
        
        Args:
            threshold: Probability threshold for buy signals (0-1)
            
        Returns:
            action: 1 for buy, 0 for hold
            confidence: Confidence score (0-1)
        """
        # Check if we have up-to-date preprocessed data
        current_time = datetime.now()
        
        # Initialize last_prediction_time if it's None
        if self.last_prediction_time is None:
            self.last_prediction_time = current_time - timedelta(minutes=self.prediction_interval)
            self.logger.info(f"Initialized last_prediction_time to {self.prediction_interval} minutes ago")

        # Check if enough time has elapsed since the last prediction
        time_since_last = (current_time - self.last_prediction_time).total_seconds() / 60
        if time_since_last < self.prediction_interval and self.last_action is not None:
            self.logger.info(f"Skipping decision, only {time_since_last:.2f} minutes since last decision")
            return self.last_action, 0

        # Update last prediction time
        self.last_prediction_time = current_time
        
        if self.preprocessed_df is None or 'ds' not in self.preprocessed_df.columns:
            self.logger.error("Cannot make decision: preprocessed data not available")
            return 0, 0
            
        try:
            self.logger.info("Getting trading decision from ML classifier")
            
            # Use the ML classifier with assume_prediction_features=True
            result = self.ml_classifier.predict(
                self.preprocessed_df, 
                prediction_model=self.ts_prediction_model,
                threshold=threshold,
                assume_prediction_features=True  # This is the new parameter
            )
            
            action = result['prediction']
            confidence = result['probability']
            
            self.logger.info(f"ML classifier decision: {'BUY' if action == 1 else 'HOLD'} with confidence {confidence:.4f}")
            self.last_action = action
            return action, confidence
                
        except Exception as e:
            self.logger.error(f"Error getting decision: {e}", exc_info=True)
            return 0, 0
        
    def fix_ml_scaler_issue(self):
        """
        Fix the ML classifier's feature scaler issue by fitting it with actual data.
        This should be called after loading the ML model and preprocessing data.
        """
        try:
            self.logger.info("Attempting to fix ML classifier scaler issue with actual data")
            
            # Check if ML classifier has a feature_scaler attribute
            if not hasattr(self.ml_classifier, 'feature_scaler'):
                self.logger.error("ML classifier has no feature_scaler attribute")
                return False
                
            # Check if we have preprocessed data
            if self.preprocessed_df is None or len(self.preprocessed_df) < self.ml_classifier.lookback_period:
                self.logger.warning("Not enough preprocessed data to fit scaler. Need at least lookback_period rows.")
                return False
                
            # Extract the features in the expected order
            expected_features = ['volume', 'trade_count', 'y', 'SMA_20_normalized',
                            'SMA_50_normalized', 'EMA_20_normalized',
                            'BBL_20_2.0_normalized', 'BBM_20_2.0_normalized',
                            'BBU_20_2.0_normalized', 'RSI_normalized',
                            'MACD_normalized', 'MACD_signal_normalized',
                            'MACD_hist_normalized']
            
            # Verify all expected features are present
            missing_features = [col for col in expected_features if col not in self.preprocessed_df.columns]
            if missing_features:
                self.logger.error(f"Cannot fit scaler: Missing required features: {missing_features}")
                return False
            
            # Create feature vectors for all available data (flattened for each lookback period)
            feature_vectors = []
            
            # Iterate through each possible window of lookback_period length
            for i in range(len(self.preprocessed_df) - self.ml_classifier.lookback_period + 1):
                window = self.preprocessed_df.iloc[i:i+self.ml_classifier.lookback_period]
                feature_vector = window[expected_features].values.flatten()
                feature_vectors.append(feature_vector)
            
            # Convert to numpy array
            feature_matrix = np.array(feature_vectors)
            
            # Fit the scaler with actual data
            self.ml_classifier.feature_scaler.fit(feature_matrix)
            
            self.logger.info(f"Successfully fitted scaler with {len(feature_vectors)} data points")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to fix ML classifier scaler issue: {e}")
            return False
                    
    def manage_trades(self):
        """Manage trades based on ML decisions"""
        self.logger.info("Managing trades...")
        
        # Check if we have a new action
        if self.last_action is None:
            self.logger.info("No action available, skipping trade management")
            return
            
        action = self.last_action
        self.logger.info(f"Current action: {'BUY' if action == 1 else 'HOLD'}")
        
        # If action is to buy
        if action == 1:
            self.logger.info("Buy signal detected")
            
            # Check if we've reached the maximum number of active trades
            if len(self.active_trades) >= self.max_trades:
                self.logger.warning(f"Maximum number of trades ({self.max_trades}) already reached, skipping buy signal")
                return
                    
            # Check if we already have a position for this symbol
            if not self.trading_client:
                self.logger.warning("No trading client available, cannot place order")
                return
                
            try:
                # Check for existing positions
                positions = self.trading_client.get_all_positions()
                symbol_positions = [p for p in positions if p.symbol == self.symbol]
                
                if not symbol_positions:
                    self.logger.info(f"No existing position for {self.symbol}, placing buy order")
                    order = self.place_order('buy')
                    
                    if order:
                        self.process_new_order(order)
                    else:
                        self.logger.error("Failed to place buy order")
                else:
                    self.logger.info(f"Position already exists for {self.symbol}, skipping buy order")
                    
            except Exception as e:
                self.logger.error(f"Error managing trades: {e}", exc_info=True)
            
            # Reset action after processing
            self.last_action = None
        else:
            self.logger.info("No buy signal, maintaining current positions")
            
    def process_new_order(self, order):
        """Process a new order that was successfully placed"""
        self.logger.info(f"Processing new order {order.id}")
        
        try:
            # Get current price
            current_price = self.get_current_price()
            
            if current_price is None:
                self.logger.error("Could not get current price for new order")
                return
                
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            
            # Generate a unique key for this trade
            trade_key = f"{datetime.now().isoformat()}-{order.id}"
            
            # Get trade size from order if possible
            try:
                if hasattr(order, 'notional'):
                    trade_amount = float(order.notional)
                elif hasattr(order, 'qty'):
                    trade_amount = float(order.qty) * current_price
                else:
                    # Get account information to calculate trade size
                    account = self.trading_client.get_account()
                    trade_amount = float(account.cash) * self.position_size_pct
            except:
                # Fallback: get account information to calculate trade size
                account = self.trading_client.get_account()
                trade_amount = float(account.cash) * self.position_size_pct
            
            # Store trade information
            self.active_trades[trade_key] = {
                'order_id': order.id,
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'symbol': self.symbol,
                'size': trade_amount
            }
            
            # Store trade in database if available
            if self.db_handler:
                db_trade_id = self.db_handler.insert_trade(
                    order_id=order.id,
                    symbol=self.symbol,
                    entry_time=datetime.now(),
                    entry_price=current_price,
                    size=trade_amount,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                if db_trade_id:
                    self.logger.info(f"Trade recorded in database with ID: {db_trade_id}")
                else:
                    self.logger.warning("Failed to record trade in database")
            
            self.logger.info(f"New trade recorded: Entry price: ${current_price:.2f}, Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}, Size: ${trade_amount:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing new order: {e}", exc_info=True)
                        
    def place_order(self, side, qty=None):
        """
        Place an order using the trading API
        
        Args:
            side: 'buy' or 'sell'
            qty: Optional quantity (uses percentage of available cash if None)
            
        Returns:
            Order object or None if failed
        """
        if not self.trading_client:
            self.logger.warning("No trading client available, cannot place real order")
            return None
            
        try:
            # Prepare market order request
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create the order request
            if qty is None and side.lower() == 'buy':
                # Calculate position size based on percentage of available cash
                account = self.trading_client.get_account()
                available_cash = float(account.cash)
                trade_amount = available_cash * self.position_size_pct
                
                self.logger.info(f"Placing {side} order for {self.position_size_pct*100}% of available cash (${trade_amount:.2f})")
                
                # Use notional amount
                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                    notional=trade_amount
                )
            else:
                # Use specified quantity
                order_data = MarketOrderRequest(
                    symbol=self.symbol,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                    qty=qty
                )
                    
            # Submit the order
            order = self.trading_client.submit_order(order_data)
            self.logger.info(f"Order placed: {side} {self.symbol}, Order ID: {order.id}")
            return order
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}", exc_info=True)
            return None
        
    def get_current_price(self):
        """
        Get the current price of the symbol
        
        Returns:
            float: Current price or None if error
        """
        try:
            client = CryptoHistoricalDataClient()
            
            # Try to get recent bar data
            request_params = CryptoBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now(timezone.utc) - timedelta(minutes=5),
                end=datetime.now(timezone.utc)
            )
            
            bars = client.get_crypto_bars(request_params)
            
            # If no bars, try quotes
            if bars is None or bars.df.empty:
                self.logger.debug(f"No recent bar data for {self.symbol}, trying quotes")
                quote_request = CryptoLatestQuoteRequest(symbol_or_symbols=self.symbol)
                quotes = client.get_crypto_latest_quote(quote_request)
                
                if self.symbol in quotes:
                    quote = quotes[self.symbol]
                    # Use mid price (average of bid and ask)
                    price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                    self.logger.debug(f"Current price from quote: ${price:.2f}")
                    return price
                else:
                    self.logger.error(f"No quote data found for {self.symbol}")
                    return None
            
            # Process bar data
            df = bars.df
            
            # Handle MultiIndex if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            # Get the most recent close price
            if len(df) > 0 and 'close' in df.columns:
                price = df['close'].iloc[-1]
                self.logger.debug(f"Current price from bars: ${price:.2f}")
                return price
            else:
                self.logger.error(f"No price data found in bars for {self.symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}", exc_info=True)
            return None

    def check_active_trades(self):
        """Check active trades for take profit or stop loss triggers"""
        if not self.active_trades:
            return
            
        self.logger.info(f"Checking {len(self.active_trades)} active trades")
        
        # Get current price
        current_price = self.get_current_price()
        
        if current_price is None:
            self.logger.error("Could not get current price to check trades")
            return
            
        # Check each active trade
        for trade_key, trade in list(self.active_trades.items()):
            self.logger.debug(f"Checking trade {trade_key}")
            
            # Check take profit
            if current_price >= trade['take_profit']:
                self.logger.info(f"Take profit triggered for {self.symbol} at ${current_price:.2f}")
                
                if self.trading_client and not trade.get('simulated', False):
                    # Get position information
                    positions = self.trading_client.get_all_positions()
                    position = next((p for p in positions if p.symbol == self.symbol), None)
                    
                    if position:
                        order = self.place_order('sell', qty=float(position.qty))
                        if order:
                            profit = (current_price - trade['entry_price']) / trade['entry_price'] * trade['size']
                            self.record_trade_result(trade_key, current_price, profit, 'Win')
                    else:
                        self.logger.warning(f"No position found for {self.symbol} when take profit was triggered")
                        del self.active_trades[trade_key]
                else:
                    # Simulated trade
                    profit = (current_price - trade['entry_price']) / trade['entry_price'] * trade['size']
                    self.record_trade_result(trade_key, current_price, profit, 'Win')
                    
            # Check stop loss
            elif current_price <= trade['stop_loss']:
                self.logger.info(f"Stop loss triggered for {self.symbol} at ${current_price:.2f}")
                
                if self.trading_client and not trade.get('simulated', False):
                    # Get position information
                    positions = self.trading_client.get_all_positions()
                    position = next((p for p in positions if p.symbol == self.symbol), None)
                    
                    if position:
                        order = self.place_order('sell', qty=float(position.qty))
                        if order:
                            loss = (trade['entry_price'] - current_price) / trade['entry_price'] * trade['size']
                            self.record_trade_result(trade_key, current_price, -loss, 'Loss')
                    else:
                        self.logger.warning(f"No position found for {self.symbol} when stop loss was triggered")
                        del self.active_trades[trade_key]
                else:
                    # Simulated trade
                    loss = (trade['entry_price'] - current_price) / trade['entry_price'] * trade['size']
                    self.record_trade_result(trade_key, current_price, -loss, 'Loss')

    def record_trade_result(self, trade_key, exit_price, profit_loss, result):
        """Record the result of a completed trade"""
        if trade_key not in self.active_trades:
            self.logger.error(f"Trade key {trade_key} not found in active trades")
            return
            
        trade = self.active_trades[trade_key]
        
        # Create trade record
        trade_record = {
            'symbol': self.symbol,
            'entry_time': trade['entry_time'],
            'exit_time': datetime.now(),
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'size': trade['size'],
            'stop_loss': trade['stop_loss'],
            'take_profit': trade['take_profit'],
            'profit_loss': profit_loss,
            'result': result,
            'simulated': trade.get('simulated', False)
        }
        
        # Update trade in database if available
        if self.db_handler and 'order_id' in trade:
            db_updated = self.db_handler.update_trade_result(
                order_id=trade['order_id'],
                exit_time=datetime.now(),
                exit_price=exit_price,
                profit_loss=profit_loss,
                result=result
            )
            if db_updated:
                self.logger.info(f"Trade result recorded in database for order: {trade['order_id']}")
            else:
                self.logger.warning(f"Failed to update trade result in database for order: {trade['order_id']}")
        
        # Update wallet balance
        self.wallet_balance += profit_loss
        
        # Update performance metrics
        if result == 'Win':
            self.performance_metrics['win_count'] += 1
            self.performance_metrics['total_profit'] += profit_loss
            self.performance_metrics['largest_win'] = max(self.performance_metrics['largest_win'], profit_loss)
        else:
            self.performance_metrics['loss_count'] += 1
            self.performance_metrics['total_loss'] += profit_loss  # This is negative
            self.performance_metrics['largest_loss'] = min(self.performance_metrics['largest_loss'], profit_loss)
        
        # Add to trade history
        self.trades_history.append(trade_record)
        
        # Remove from active trades
        del self.active_trades[trade_key]
        
        self.logger.info(f"Trade completed: {result} with {'profit' if profit_loss > 0 else 'loss'} of ${abs(profit_loss):.2f}")
        self.logger.info(f"Current wallet balance: ${self.wallet_balance:.2f}")
        
        # Log updated performance metrics
        total_trades = self.performance_metrics['win_count'] + self.performance_metrics['loss_count']
        win_rate = self.performance_metrics['win_count'] / total_trades if total_trades > 0 else 0
        
        self.logger.info(f"Performance metrics: Win rate: {win_rate*100:.2f}%, Total profit: ${self.performance_metrics['total_profit']:.2f}")
        
        # Save trade history periodically
        if len(self.trades_history) % 5 == 0:
            self.save_trade_history()

    def get_account_info(self):
        """Get updated account information"""
        if not self.trading_client:
            self.logger.info("No trading client available, showing simulated account info")
            self.logger.info(f"Simulated Wallet Balance: ${self.wallet_balance:.2f}")
            self.logger.info(f"Active Trades: {len(self.active_trades)}")
            
            # Show active trade details
            for key, trade in self.active_trades.items():
                current_price = self.get_current_price()
                if current_price:
                    unrealized_pl = (current_price - trade['entry_price']) / trade['entry_price'] * trade['size']
                    self.logger.info(f"  Trade: Entry: ${trade['entry_price']:.2f}, Current: ${current_price:.2f}, Unrealized P/L: ${unrealized_pl:.2f}")
            
            return None
        
        try:
            self.account = self.trading_client.get_account()
            self.logger.info(f"Account Balance: ${float(self.account.cash):.2f}")
            self.logger.info(f"Account Value: ${float(self.account.equity):.2f}")
            self.logger.info(f"P&L Today: ${float(self.account.equity) - float(self.account.last_equity):.2f}")
            
            # Get all positions
            positions = self.trading_client.get_all_positions()
            position = next((p for p in positions if p.symbol == self.symbol), None)
            
            if position:
                self.logger.info(f"\nCurrent Position for {self.symbol}:")
                self.logger.info(f"Quantity: {position.qty} shares")
                self.logger.info(f"Entry Price: ${position.avg_entry_price}")
                self.logger.info(f"Current Price: ${position.current_price}")
                self.logger.info(f"P&L: ${float(position.unrealized_pl):.2f}")
                
                # Print stop loss and take profit if available
                for trade_key, trade in self.active_trades.items():
                    self.logger.info(f"Stop Loss: ${trade['stop_loss']:.2f}")
                    self.logger.info(f"Take Profit: ${trade['take_profit']:.2f}")
            else:
                self.logger.info(f"\nNo open position for {self.symbol}")
                
            return self.account
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}", exc_info=True)
            return None

    def save_trade_history(self, filepath=None):
        """Save trade history to a CSV file"""
        if not self.trades_history:
            self.logger.warning("No trade history to save")
            return
            
        if filepath is None:
            filepath = f"trade_history_{self.symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            
        try:
            df = pd.DataFrame(self.trades_history)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Trade history saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}", exc_info=True)

    def load_database_trades(self):
        """
        Load active trades from the database into memory.
        This is useful when restarting the bot to recover the state.
        
        Returns:
            int: Number of trades loaded
        """
        if not self.db_handler:
            self.logger.warning("No database handler available, cannot load trades")
            return 0
            
        try:
            # Get active trades from database
            db_trades = self.db_handler.get_active_trades(self.symbol)
            
            # Add each trade to active_trades
            count = 0
            for db_trade in db_trades:
                trade_key = f"{db_trade['entry_time'].isoformat()}-{db_trade['order_id']}"
                
                self.active_trades[trade_key] = {
                    'order_id': db_trade['order_id'],
                    'entry_time': db_trade['entry_time'],
                    'entry_price': float(db_trade['entry_price']),
                    'stop_loss': float(db_trade['stop_loss']),
                    'take_profit': float(db_trade['take_profit']),
                    'symbol': db_trade['symbol'],
                    'size': float(db_trade['size'])
                }
                count += 1
                
            self.logger.info(f"Loaded {count} active trades from database")
            return count
            
        except Exception as e:
            self.logger.error(f"Error loading trades from database: {e}", exc_info=True)
            return 0

    def generate_performance_report(self, show_plot=True, use_database=False):
        """
        Generate a performance report and optionally display a plot.
        
        Args:
            show_plot: Whether to show plots
            use_database: Whether to use database for performance data
            
        Returns:
            Dictionary of performance metrics
        """
        # If using database and we have a database handler
        if use_database and self.db_handler:
            try:
                # Get metrics from database
                metrics = self.db_handler.get_performance_metrics(self.symbol)
                
                # Print report
                self.logger.info("\n\n===== DATABASE PERFORMANCE REPORT =====")
                self.logger.info(f"Symbol: {self.symbol}")
                self.logger.info(f"Total Trades: {metrics['total_trades']}")
                self.logger.info(f"Winning Trades: {metrics['win_count']} ({metrics['win_rate']*100:.2f}%)")
                self.logger.info(f"Losing Trades: {metrics['loss_count']} ({(1-metrics['win_rate'])*100:.2f}%)")
                self.logger.info(f"Largest Win: ${float(metrics['largest_win']):.2f}")
                self.logger.info(f"Largest Loss: ${float(metrics['largest_loss']):.2f}")
                self.logger.info(f"Net Profit/Loss: ${float(metrics['total_profit_loss']):.2f}")
                self.logger.info("===============================\n")
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error generating database performance report: {e}")
                # Fall back to local history
                self.logger.info("Falling back to local trade history for report")
        
        # If not using database or fallback from error
        if not self.trades_history:
            self.logger.warning("No trade history for performance report")
            return {}
            
        # Create DataFrame from trade history
        df = pd.DataFrame(self.trades_history)
        
        # Calculate metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit_loss'] > 0])
        losing_trades = len(df[df['profit_loss'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = df[df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['profit_loss'] <= 0]['profit_loss'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(df[df['profit_loss'] > 0]['profit_loss'].sum() / df[df['profit_loss'] <= 0]['profit_loss'].sum()) if df[df['profit_loss'] <= 0]['profit_loss'].sum() != 0 else float('inf')
        
        # Calculate equity curve
        df = df.sort_values('exit_time')
        df['cumulative_pl'] = df['profit_loss'].cumsum()
        
        # Print report
        self.logger.info("\n\n===== PERFORMANCE REPORT =====")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Period: {df['entry_time'].min()} to {df['exit_time'].max()}")
        self.logger.info(f"Total Trades: {total_trades}")
        self.logger.info(f"Winning Trades: {winning_trades} ({win_rate*100:.2f}%)")
        self.logger.info(f"Losing Trades: {losing_trades} ({(1-win_rate)*100:.2f}%)")
        self.logger.info(f"Profit Factor: {profit_factor:.2f}")
        self.logger.info(f"Average Win: ${avg_win:.2f}")
        self.logger.info(f"Average Loss: ${avg_loss:.2f}")
        self.logger.info(f"Largest Win: ${df['profit_loss'].max():.2f}")
        self.logger.info(f"Largest Loss: ${df['profit_loss'].min():.2f}")
        self.logger.info(f"Net Profit/Loss: ${df['profit_loss'].sum():.2f}")
        self.logger.info(f"Final Equity: ${df['cumulative_pl'].iloc[-1]:.2f}")
        self.logger.info("===============================\n")
        
        # Plot equity curve
        if show_plot:
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(df['exit_time'], df['cumulative_pl'])
            plt.title('Equity Curve')
            plt.ylabel('Cumulative Profit/Loss ($)')
            plt.grid(True)
            
            # Plot trade outcomes
            plt.subplot(2, 1, 2)
            
            # Plot winning trades in green
            wins = df[df['profit_loss'] > 0]
            if not wins.empty:
                plt.bar(range(len(wins)), wins['profit_loss'], color='green', label='Wins')
                
            # Plot losing trades in red
            losses = df[df['profit_loss'] <= 0]
            if not losses.empty:
                plt.bar(range(len(wins), len(df)), losses['profit_loss'], color='red', label='Losses')
                
            plt.title('Trade Outcomes')
            plt.xlabel('Trade Number')
            plt.ylabel('Profit/Loss ($)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"performance_{self.symbol}_{datetime.now().strftime('%Y%m%d')}.png")
            plt.show()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_pl': df['profit_loss'].sum(),
            'final_equity': df['cumulative_pl'].iloc[-1] if not df.empty else 0
        }

    def close_position(self):
        """Close the open position for the symbol"""
        self.logger.info(f"Attempting to close position for {self.symbol}")
        
        if not self.trading_client:
            self.logger.info("No trading client, simulating position close")
            for trade_key, trade in list(self.active_trades.items()):
                current_price = self.get_current_price()
                if current_price:
                    profit_loss = (current_price - trade['entry_price']) / trade['entry_price'] * trade['size']
                    result = 'Win' if profit_loss > 0 else 'Loss'
                    self.record_trade_result(trade_key, current_price, profit_loss, result)
            self.logger.info("All simulated positions closed")
            return
            
        try:
            # Get all positions
            positions = self.trading_client.get_all_positions()
            position = next((p for p in positions if p.symbol == self.symbol), None)
            
            if not position:
                self.logger.info(f"No position to close for {self.symbol}")
                return
                
            self.logger.info(f"Closing position for {self.symbol}...")
            
            # Place a market sell order
            order = self.place_order('sell', qty=float(position.qty))
            if order:
                self.logger.info(f"Closing order placed for {self.symbol}, {position.qty} shares")
                
                # Record trade results for all active trades
                current_price = self.get_current_price() or float(position.current_price)
                for trade_key, trade in list(self.active_trades.items()):
                    profit_loss = (current_price - trade['entry_price']) / trade['entry_price'] * trade['size']
                    result = 'Win' if profit_loss > 0 else 'Loss'
                    self.record_trade_result(trade_key, current_price, profit_loss, result)
                
                self.logger.info("Position closed and all trades recorded")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}", exc_info=True)

    def cleanup(self):
        """
        Clean up resources before shutdown.
        Should be called when exiting the application.
        """
        try:
            # Generate final reports
            self.save_trade_history()
            self.generate_performance_report(use_database=True)
            
            # Close any open positions
            self.close_position()
            
            # Disconnect from database
            if self.db_handler:
                self.db_handler.disconnect()
                self.logger.info("Database connection closed")
                
            self.logger.info("Trading bot cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

    def run(self):
        """
        Run the trading bot in a continuous loop
        """
        self.logger.info(f"Starting trading bot for {self.symbol}")
        self.logger.info(f"{'PAPER' if self.paper else 'LIVE'} trading mode enabled")
        
        # Configuration for retry logic
        max_retries = 3
        retry_count = 0
        retry_delay = 10  # seconds
        
        # Before entering main loop, fetch initial data
        self.fetch_historical_data()
        
        # If we have a database handler, load active trades from database
        if self.db_handler:
            self.load_database_trades()
        
        try:
            while True:
                try:
                    current_time = datetime.now()
                    self.logger.info(f"\n--- Trading Cycle: {current_time} ---")
                    
                    # Check active trades every cycle
                    try:
                        self.check_active_trades()
                    except Exception as e:
                        self.logger.error(f"Error checking active trades: {e}", exc_info=True)
                    
                    # Fetch data every cycle
                    try:
                        self.fetch_historical_data()
                    except Exception as e:
                        self.logger.error(f"Error fetching data: {e}", exc_info=True)
                    
                    # Get decision and manage trades
                    try:
                        action, confidence = self.get_decision()
                        # If a decision was made, manage trades based on it
                        if action is not None:
                            self.manage_trades()
                    except Exception as e:
                        self.logger.error(f"Error in decision/trade cycle: {e}", exc_info=True)
                    
                    # Get account info every cycle
                    try:
                        self.get_account_info()
                    except Exception as e:
                        self.logger.error(f"Error getting account info: {e}", exc_info=True)
                    
                    self.logger.info(f"Waiting {self.check_interval * 60} seconds until next cycle...\n")
                    time.sleep(self.check_interval * 60)
                    
                    # Reset retry count on successful cycle
                    retry_count = 0
                    
                except KeyboardInterrupt:
                    self.logger.info("\nTrading bot stopped by user")
                    self.cleanup()
                    break
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                    
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Maximum retries ({max_retries}) exceeded. Stopping bot.")
                        break
                        
                    self.logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    
        finally:
            # Clean up and generate final report
            self.logger.info("Trading bot stopping, generating final report...")
            self.cleanup()