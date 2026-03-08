import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model: LinearRegression, X: pd.DataFrame) -> pd.Series:
    preds = model.predict(X)
    return pd.Series(preds, index=X.index)
