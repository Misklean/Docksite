import os
import pandas as pd
import joblib
import logging
import configparser
from datetime import datetime

# Import the updated trading bot class
from classes.trading_bot.trading_bot_class import TradingBot
# Import DatabaseHandler
from classes.database.database_handler import DatabaseHandler

def setup_logging():
    """Set up logging for the main application"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{log_dir}/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("Main")

def load_config(config_file="config.ini"):
    """Load configuration from the config file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Trading Bot Application")
    
    try:
        # Load configuration
        config = load_config()
        
        # Get Alpaca API credentials
        api_key = config['alpaca'].get('API_KEY')
        api_secret = config['alpaca'].get('API_SECRET')
        paper_trading = config['alpaca'].getboolean('PAPER_TRADING', True)
        
        # Get trading parameters
        symbol = config['trading'].get('SYMBOL', 'BTC/USD')
        position_size_pct = config['trading'].getfloat('POSITION_SIZE_PCT', 0.1)
        stop_loss_pct = config['trading'].getfloat('STOP_LOSS_PCT', 0.02)
        take_profit_pct = config['trading'].getfloat('TAKE_PROFIT_PCT', 0.02)
        max_trades = config['trading'].getint('MAX_TRADES', 5)
        prediction_interval = config['trading'].getint('PREDICTION_INTERVAL_MINUTES', 15)
        check_interval = config['trading'].getint('CHECK_INTERVAL_MINUTES', 1)
        
        # Get model paths
        ts_model_path = config['models'].get('TS_MODEL_PATH', 'ressource/models/prediction_model.pkl')
        ml_model_path = config['models'].get('ML_MODEL_PATH', 'ressource/models/trading_ml_model.joblib')
        
        # Get database config if available
        db_config = None
        if 'database' in config:
            db_config = {
                'dbname': config['database'].get('DBNAME', 'trading_bot'),
                'user': config['database'].get('USER', 'postgres'),
                'password': config['database'].get('PASSWORD', ''),
                'host': config['database'].get('HOST', 'localhost'),
                'port': config['database'].get('PORT', '5432')
            }
            logger.info(f"Database configuration loaded: {db_config['dbname']} on {db_config['host']}")
        else:
            logger.warning("No database configuration found in config file")
            
        # Load time series prediction model
        logger.info(f"Loading time series prediction model from {ts_model_path}")
        ts_prediction_model = joblib.load(ts_model_path)
        
        # Initialize trading bot
        logger.info("Initializing trading bot")
        trading_bot = TradingBot(
            ts_prediction_model=ts_prediction_model,
            ml_model_path=ml_model_path,
            symbol=symbol,
            api_key=api_key,
            api_secret=api_secret,
            paper=paper_trading,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_trades=max_trades,
            prediction_interval=prediction_interval,
            check_interval=check_interval,
            db_config=db_config  # Add database configuration
        )
        
        # Run the trading bot
        logger.info("Starting trading bot main loop")
        trading_bot.run()
    
    except KeyboardInterrupt:
        logger.info("Trading Bot stopped by user")
    except Exception as e:
        logger.error(f"Error in Trading Bot application: {e}", exc_info=True)
    finally:
        logger.info("Trading Bot application terminated")

if __name__ == "__main__":
    main()