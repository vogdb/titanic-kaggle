import pandas as pd
import pkg_resources


def load_data():
    train_filename = pkg_resources.resource_filename(__name__, 'dataset/train.csv')
    X_train = pd.read_csv(train_filename)
    y_train = X_train.Survived

    test_filename = pkg_resources.resource_filename(__name__, 'dataset/test.csv')
    X_test = pd.read_csv(test_filename)

    return X_train, y_train, X_test
