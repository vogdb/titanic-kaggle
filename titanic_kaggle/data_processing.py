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


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return X.fillna(self.most_frequent_)


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
        X = X.copy()
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
        X = X.copy()
        X['AgeBucket'] = X.Age // 15 * 15
        X['RelativesOnBoard'] = X.SibSp + X.Parch
        # is a reverend, they all die :(
        X['Rev'] = X.Name.str.contains('rev.|Rev.')
        return X
