import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import joblib


def load_and_process(filepath):

    print("Loading data from", filepath)

    data = pd.read_csv(filepath)
    data['date'] = pd.to_datetime(data['date'])
    data_hourly = (
            data.set_index('date')
            .resample('H').mean()
            .dropna()
            .reset_index()
    )
    return data_hourly

def lagged_features(df, lag_value = 1):
    """
    Create a new dataframe with new features. For the moment we just considerr lagged features
    """
    df = df.copy()
    df['value_lag'] = df['value'].shift(lag_value).dropna().reset_index(drop=True)
    return df

def feature_engineering(df):
    """
    Create a new dataframe with new features. For the moment we just considerr lagged features
    """
    df = df.copy()
    df = lagged_features(df)
    # df['value_lag_diff'] = df['value'] - df['value_lag']
    # df['value_lag_diff_pct'] = df['value_lag_diff'] / df['value_lag']
    return df
   
def train_model(df):
    """
    Split the data into training and testing sets,
    then train a simple regression model using a pipeline.
    """
    X = df[["lag_pm25"]]
    y = df["value"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained. Test R^2 score: {score:.2f}")
    return pipeline


if __name__ == '__main__':
    data = load_and_process('data_Machala.csv')
    data = feature_engineering(data)
    model = train_model(data)
    print("Model trained and saved.")
    joblib.dump(model, 'model.joblib')
    print("Model saved to model.joblib")