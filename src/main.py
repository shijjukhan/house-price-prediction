import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, preprocess_data
from exploratory_data_analysis import plot_pairwise_relationships, plot_correlation_heatmap
from model_training import train_model, evaluate_model
from model_optimization import optimize_model

def main():
    # Load and preprocess data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Perform EDA
    plot_pairwise_relationships(data)
    plot_correlation_heatmap(data)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Optimize model
    optimize_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()

