# Import necessary libraries
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(stocks, start_date, end_date):
    # Download data for stocks
    data = yf.download(stocks, start=start_date, end=end_date)
    # Drop rows with missing values
    data = data.dropna()
    # Engineer additional features
    data = engineer_features(data, stocks)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_imputed)
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data_norm, test_size=0.2)
    return train_data, test_data

def engineer_features(data, stocks):
    # Check the shape of the data DataFrame
    num_rows, num_columns = data.shape
    print(f"The data DataFrame has {num_rows} rows and {num_columns} columns.")
    
    # Print out the contents of the data DataFrame
    print(data)
    
    # Calculate moving average (MA5) for each stock
    for stock in stocks:
        data[f'{stock} MA5'] = data['Close'][stock].rolling(window=5).mean()
        
    # Calculate moving average (MA10) for each stock
    for stock in stocks:
        data[f'{stock} MA10'] = data['Close'][stock].rolling(window=10).mean()
        
    # Calculate moving average (MA20) for each stock
    for stock in stocks:
        data[f'{stock} MA20'] = data['Close'][stock].rolling(window=20).mean()
        
    # Calculate relative strength index (RSI) for each stock
    for stock in stocks:
        delta = data['Close'][stock].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data[f'{stock} RSI'] = rsi
        
    return data

def create_and_train_model(train_data):
    # Split data into features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    # Create and train model (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

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

def main():
    # Read tickers from file
    with open("all_tickers.txt", "r") as file:
        stocks = [ticker.strip() for ticker in file.readlines()]
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
