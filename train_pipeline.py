import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

import pipeline
import config



def run_training():
    """Train the model."""

    data = pd.read_csv('data/train.csv')
    data['GrLivAndBsmtArea'] = data['TotalBsmtSF'] + data['GrLivArea']

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1),
        data[config.TARGET],
        test_size=0.2,
        random_state=1)

    pipeline.total_pipeline.fit(X_train, y_train)
    joblib.dump(pipeline.total_pipeline, 'model/house_prices_pipeline.pkl')


if __name__ == '__main__':
    run_training()