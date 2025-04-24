import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import joblib
import logging
import argparse

# Add parent directory to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the trading ML classifier
from classes.ml_classifier.trading_ml_classifier import TradingMLClassifier


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Train ML trading model')
    parser.add_argument('--load_existing', action='store_true', help='Load existing dataset if available')
    parser.add_argument('--skip_backtest', action='store_true', help='Skip backtesting')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--model_path', type=str, default='ressource/models/trading_ml_model.joblib', 
                        help='Path to save/load the model')
    parser.add_argument('--dataset_path', type=str, default='ressource/dataset/ml_dataset.npz',
                        help='Path to save/load the dataset')
    
    args = parser.parse_args()

    print(args)
    
    # Load data
    logging.info("Loading data...")
    preprocessed_path = 'ressource/dataset/data_2024_hour_preprocessed.csv'
    original_path = 'ressource/dataset/data_2024_original.csv'
    
    if not os.path.exists(preprocessed_path) or not os.path.exists(original_path):
        logging.error(f"Data files not found: {preprocessed_path} or {original_path}")
        return
    
    hour_preprocessed_df = pd.read_csv(preprocessed_path)
    original_df = pd.read_csv(original_path)
    
    logging.info(f"Loaded preprocessed data with shape: {hour_preprocessed_df.shape}")
    logging.info(f"Loaded original data with shape: {original_df.shape}")
    
    # Initialize the classifier
    classifier = TradingMLClassifier(
        preprocessed_df=hour_preprocessed_df,
        original_df=original_df
    )
    
    # Create dataset
    logging.info("Creating dataset...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.create_dataset(
            stop_loss_pct=0.02,
            take_profit_pct=0.02,
            validation_size=0.2,
            test_size=0.2,
            load_if_exists=args.load_existing,
            dataset_path=args.dataset_path
        )
        logging.info("Dataset created successfully")
    except Exception as e:
        logging.error(f"Error creating dataset: {e}")
        return
    
    # Train models
    logging.info("Training models...")
    try:
        best_model = classifier.train_models(class_weight='balanced')
        logging.info(f"Training complete. Best model: {type(best_model).__name__}")
    except Exception as e:
        logging.error(f"Error training models: {e}")
        return
    
    # Evaluate model
    logging.info("Evaluating model...")
    try:
        metrics = classifier.evaluate_model()
        logging.info(f"Evaluation metrics: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, " +
                     f"Recall={metrics['recall']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
    
    # Analyze feature importance
    logging.info("Analyzing feature importance...")
    try:
        importance_df = classifier.feature_importance()
        if importance_df is not None:
            logging.info(f"Top 5 important features: {importance_df.head(5)['feature'].tolist()}")
            plt.savefig('ressource/feature_importance.png')
            plt.close()
    except Exception as e:
        logging.error(f"Error analyzing feature importance: {e}")
    
    # Backtest strategy if not skipped
    if not args.skip_backtest:
        logging.info("Backtesting strategy...")
        try:
            backtest_df, trades_df = classifier.backtest_strategy(
                initial_balance=10000,
                position_size_pct=0.1
            )
            if len(trades_df) > 0:
                plt.savefig('ressource/equity_curve.png')
                plt.close()
                logging.info(f"Backtest complete. Final balance: ${trades_df.iloc[-1]['balance']:.2f}")
                trades_df.to_csv('ressource/backtest_trades.csv', index=False)
            else:
                logging.warning("No trades executed in backtest")
        except Exception as e:
            logging.error(f"Error backtesting strategy: {e}")
    
    # Save model if requested
    if args.save_model:
        logging.info(f"Saving model to {args.model_path}...")
        try:
            classifier.save_model(args.model_path)
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

if __name__ == "__main__":
    main()