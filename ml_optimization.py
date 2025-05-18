from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_capacity_model(features, capacities, model_path):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, capacities)
    joblib.dump(model, model_path)
    print(f"Capacity model saved to {model_path}")

def predict_capacity(features, model_path):
    model = joblib.load(model_path)
    features = np.array(features).reshape(1, -1)
    return int(model.predict(features)[0])


def train_model(X, y, model_path):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def predict_threshold(features, model_path):
    model = joblib.load(model_path)
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]
