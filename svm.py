import os

import numpy as np
import pandas as pd
from scipy.stats import expon, reciprocal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from estimator_serialize import EstimatorSerialize


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self

    def transform(self, X, y=None):
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
        X['AgeBucket'] = X.Age // 15 * 15
        X['RelativesOnBoard'] = X.SibSp + X.Parch
        # is a reverend, they all die :(
        X['Rev'] = X.Name.str.contains('rev.|Rev.')
        return X


def create_prepare_pipeline():
    embarked_transformer = Pipeline([
        ('fill na', MostFrequentImputer()),
        ('onehot', OneHotEncoder(sparse=False)),
    ])

    column_transformer = ColumnTransformer([
        ('pass', 'passthrough', ['RelativesOnBoard', 'Rev']),
        ('fare', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), ['Fare']),
        ('age', StandardScaler(), ['Age']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'AgeBucket']),
        ('embarked', embarked_transformer, ['Embarked']),
    ], remainder='drop')

    return Pipeline([
        ('age', AgeImputer()),
        ('new_columns', NewColumnsTransformer()),
        ('prepare', column_transformer),
    ])


df = pd.read_csv(os.path.join('dataset', 'train.csv'))
y = df.Survived

prepare_pipeline = create_prepare_pipeline()
X = prepare_pipeline.fit_transform(df)

svm_clf = SVC()
param_distribs = {
    'kernel': ['linear', 'rbf'],
    'C': reciprocal(20, 200000),
    'gamma': expon(scale=1.0),
}
search = RandomizedSearchCV(
    svm_clf, param_distributions=param_distribs,
    n_iter=20, cv=5, scoring='accuracy', n_jobs=4, random_state=42
)
search.fit(X, y)
EstimatorSerialize.save_estimator('svm', search.best_estimator_)
