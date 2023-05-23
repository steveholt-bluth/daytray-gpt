# Import necessary libraries
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

def preprocess_data(stocks, start_date, end_date):
    # Download data for stocks
    data = yf.download(stocks, start=start_date, end=end_date)
    # Drop rows with missing values
    data = data.dropna()
    # Normalize data
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data_norm, test_size=0.2)
    return train_data, test_data

# Define function to create and train model

def create_and_train_model(train_data):
    # Split data into features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    # Create and train model (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Define function to make predictions

def make_predictions(model, test_data):
    # Split data into features and target
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    # Make predictions
    predictions = model.predict(X_test)
    # Calculate accuracy (using Mean Squared Error as a metric)
    mse = mean_squared_error(y_test, predictions)
    accuracy = 1 - mse
    return predictions, accuracy

# Define main function

def main():
    # Define stocks to trade
    stocks = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT']
    # Define start and end dates
    start_date = '2010-01-01'
    end_date = '2023-05-23'
    # Preprocess data
    train_data, test_data = preprocess_data(stocks, start_date, end_date)
    # Create and train model
    model = create_and_train_model(train_data)
    # Make predictions
    predictions, accuracy = make_predictions(model, test_data)
    # Print accuracy
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    main()
