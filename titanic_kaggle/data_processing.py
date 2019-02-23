from typing import List, Dict, Tuple

import inflection
import numpy as np
import pandas as pd
import pkg_resources
from sklearn.base import BaseEstimator, TransformerMixin


def load_data():
    train_filename = pkg_resources.resource_filename(__name__, 'dataset/train.csv')
    X_train = pd.read_csv(train_filename)
    y_train = X_train.Survived

    test_filename = pkg_resources.resource_filename(__name__, 'dataset/test.csv')
    X_test = pd.read_csv(test_filename)

    return X_train, y_train, X_test


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._mean_mrs_age = np.floor(
            X.loc[X.Age.notnull() & X.Name.str.contains('Mrs')]['Age'].mean()
        )
        self._mean_age_greater_15 = np.floor(
            X.loc[(X.Age >= 15) & (X.Parch == 0) & (X.SibSp == 0)]['Age'].mean()
        )
        self._mean_age_male_3class = np.floor(
            X.loc[(X.Sex == 'male') & (X.Pclass == 3)]['Age'].mean()
        )
        self._mean_age_female_3class = np.floor(
            X.loc[(X.Sex == 'female') & (X.Pclass == 3)]['Age'].mean()
        )
        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)
        X.loc[X.Age.isnull() & X.Name.str.contains('Mrs'), 'Age'] = self._mean_mrs_age
        # there is 400 records of people with age >= 15 and 0 relatives vs 4 records of age < 15 and 0 relatives
        X.loc[X.Age.isnull() & (X.Parch == 0) & (X.SibSp == 0), 'Age'] = self._mean_age_greater_15
        # the remaining null Age contains only 3 class
        X.loc[X.Age.isnull() & (X.Sex == 'male') & (X.Pclass == 3), 'Age'] = self._mean_age_male_3class
        X.loc[X.Age.isnull() & (X.Sex == 'female') & (X.Pclass == 3), 'Age'] = self._mean_age_female_3class

        return X


class NewColumnsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy(deep=True)
        X['AgeBucket'] = to_agebucket(X)
        X['RelativesOnBoard'] = X.SibSp + X.Parch
        # is a reverend, they all die :(
        X['Rev'] = X.Name.str.contains('rev.|Rev.')
        return X


def to_agebucket(X):
    return X.Age.astype(np.int32) // 15 * 15


def do_num_bucket(data: pd.DataFrame, attrs: List, suffix: str = '_bucket', params: Dict = None) \
        -> Tuple[pd.DataFrame, List]:
    new_attrs = [attr + suffix for attr in attrs]
    mask = data[attrs].notnull()
    data_tmp = (data[attrs].fillna(0) // params.len * params.len).astype(np.int32)
    data[new_attrs] = data_tmp.where(mask)

    if params.drop:
        data = data.drop(attrs, axis=1)
    return data, new_attrs


def snake_case_column_names(data: pd.DataFrame):
    old_column_names = data.columns
    new_column_names = [inflection.underscore(column) for column in old_column_names]
    return data.rename(columns=dict(zip(old_column_names, new_column_names)))


if __name__ == '__main__':
    X_train, y_train, X_test = load_data()
    X_train = snake_case_column_names(X_train)
    print(X_train.columns)
    X_train, _ = do_num_bucket(X_train, ['age', 'fare'], params=dict(len=15, drop=True))
    print(X_train.head())
