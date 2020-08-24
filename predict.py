import pandas as pd

import joblib
import config


def make_prediction(input_data):
    _pipe = joblib.load(filename='model/house_prices_pipeline.pkl')

    results = _pipe.predict(input_data)

    return results


if __name__ == '__main__':

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv('data/train.csv')
    data['GrLivAndBsmtArea'] = data['TotalBsmtSF'] + data['GrLivArea']

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.2,
        random_state=1)

    pred = make_prediction(X_test)

    # determine mse and rmse
    print('test mse: {}'.format(int(
        mean_squared_error(y_test, pred))))
    print('test rmse: {}'.format(int(
        np.sqrt(mean_squared_error(y_test, pred)))))
    print('test r2: {}'.format(
        r2_score(y_test, pred)))


    test = pd.read_csv('data/test.csv').set_index('Id')
    test['GrLivAndBsmtArea'] = test['TotalBsmtSF'] + test['GrLivArea']
    pred_lb = make_prediction(test)
    pred_lb = pd.DataFrame(pred_lb, columns = ['SalePrice'], index=test.index).reset_index()
    pred_lb.to_csv('data/submission.csv',index=False)


# %%

