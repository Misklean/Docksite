"""
Script to backtest a trained trading model on historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the trading ML classifier
from classes.ml_classifier.trading_ml_classifier import TradingMLClassifier

def main():
    parser = argparse.ArgumentParser(description='Backtest a trading ML model')
    parser.add_argument('--preprocessed-data', type=str, default='ressource/dataset/data_2023_hour_preprocessed.csv',
                        help='Path to preprocessed data CSV')
    parser.add_argument('--original-data', type=str, default='ressource/dataset/data_2023_original.csv',
                        help='Path to original price data CSV')
    parser.add_argument('--model-path', type=str, default='ressource/models/trading_ml_classifier_2perc.joblib',
                        help='Path to the trained model')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for backtest (default: 10000)')
    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as a fraction of balance (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='ressource/backtests',
                        help='Directory to save backtest results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for trading signals (default: 0.5)')
    args = parser.parse_args()

    print(f"Starting backtest at {datetime.now()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the classifier
    classifier = TradingMLClassifier()
    
    # Load trained model
    print(f"Loading model from {args.model_path}")
    model = classifier.load_model(args.model_path)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # For backtesting, we need to load the same data used for training
    print(f"Loading data for backtesting")
    preprocessed_df = pd.read_csv(args.preprocessed_data)
    original_df = pd.read_csv(args.original_data)
    
    # Set dataframes for backtesting
    classifier.preprocessed_df = preprocessed_df
    classifier.original_df = original_df
    
    # Create dataset for backtesting (this will use the same splits as training)
    dataset_path = os.path.join(os.path.dirname(args.model_path), 'training_dataset.npz')
    print(f"Loading dataset from {dataset_path}")
    classifier.create_dataset(load_if_exists=True, dataset_path=dataset_path)
    
    # Run backtest
    print(f"Running backtest with initial balance ${args.initial_balance:.2f} and position size {args.position_size * 100:.2f}%")
    backtest_df, trades_df = classifier.backtest_strategy(
        initial_balance=args.initial_balance,
        position_size_pct=args.position_size
    )
    
    # Save backtest results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_file = os.path.join(args.output_dir, f"backtest_results_{timestamp}.csv")
    trades_file = os.path.join(args.output_dir, f"trades_{timestamp}.csv")
    
    backtest_df.to_csv(backtest_file, index=False)
    if not trades_df.empty:
        trades_df.to_csv(trades_file, index=False)
    
    print(f"Backtest results saved to {backtest_file}")
    if not trades_df.empty:
        print(f"Trade details saved to {trades_file}")
    
    # Plot backtest results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Equity curve
    if not trades_df.empty:
        plt.subplot(2, 1, 1)
        plt.plot(trades_df['exit_time'], trades_df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.grid(True)
    
    # Plot 2: Trade signals on price chart
    plt.subplot(2, 1, 2)
    
    # Get timestamp range from backtest_df
    if not backtest_df.empty:
        start_time = backtest_df['timestamp'].min()
        end_time = backtest_df['timestamp'].max()
        
        # Extract price data for this time range
        price_data = original_df[
            (original_df['timestamp'] >= start_time) & 
            (original_df['timestamp'] <= end_time)
        ]
        
        plt.plot(price_data['timestamp'], price_data['close'], 'b-', label='Price')
        
        # Plot buy signals
        buy_signals = backtest_df[backtest_df['predicted_signal'] == 1]
        if not buy_signals.empty:
            plt.scatter(buy_signals['timestamp'], 
                       buy_signals['entry_price'], 
                       color='g', marker='^', s=100, label='Buy Signal')
        
        # Plot trades
        if not trades_df.empty:
            # Plot winning trades
            wins = trades_df[trades_df['result'] == 'Win']
            if not wins.empty:
                plt.scatter(wins['exit_time'], 
                          wins['exit_price'], 
                          color='g', marker='o', s=100, label='Take Profit')
            
            # Plot losing trades
            losses = trades_df[trades_df['result'] == 'Loss']
            if not losses.empty:
                plt.scatter(losses['exit_time'], 
                          losses['exit_price'], 
                          color='r', marker='x', s=100, label='Stop Loss')
        
        plt.title('Trading Signals on Price Chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    
    # Save figure
    plot_file = os.path.join(args.output_dir, f"backtest_plot_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Backtest plot saved to {plot_file}")
    
    # Show evaluation metrics at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nModel performance at different probability thresholds:")
    for threshold in thresholds:
        eval_results = classifier.evaluate_model(threshold=threshold)
        print(f"Threshold {threshold:.2f} - F1: {eval_results['f1']:.4f}, Precision: {eval_results['precision']:.4f}, Recall: {eval_results['recall']:.4f}")

    print(f"Backtest completed at {datetime.now()}")

if __name__ == "__main__":
    main()