from scipy.stats import expon, reciprocal, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import numpy as np
from titanic_kaggle.data_processing import MostFrequentImputer, AgeImputer, NewColumnsTransformer, load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize


def create_prepare_pipeline():
    column_transformer = ColumnTransformer([
        ('pass', 'passthrough', ['RelativesOnBoard', 'Rev']),
        ('fare', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), ['Fare']),
        ('age', StandardScaler(), ['Age']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex']),
        ('embarked', Pipeline([
            ('fill na', MostFrequentImputer()),
            ('onehot', OneHotEncoder(sparse=False)),
        ]), ['Embarked']),
    ], remainder='drop')

    return Pipeline([
        ('age', AgeImputer()),
        ('new_columns', NewColumnsTransformer()),
        ('prepare', column_transformer),
    ])


def main():
    X_train, y_train, X_test = load_data()

    # param_distribs = {
    #     'kernel': ['rbf', 'poly', 'sigmoid'],
    #     'C': reciprocal(.1, 100),
    #     'gamma': reciprocal(.001, 1),
    # }
    # search = RandomizedSearchCV(
    #     SVC(), param_distributions=param_distribs,
    #     n_iter=50, cv=5, scoring='accuracy', n_jobs=-1
    # )
    param_grid = {
        'kernel': ['rbf'],
        'gamma': np.arange(0.01, 0.11, 0.01),
        'C': np.arange(1, 11),
    }
    search_cv = GridSearchCV(
        SVC(probability=True), param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    prepare_pipeline = create_prepare_pipeline()
    search_pipeline = Pipeline([
        ('prepare', prepare_pipeline),
        ('search', search_cv),
    ])
    search_pipeline.fit(X_train, y_train)

    best_pipeline = Pipeline([
        ('prepare', prepare_pipeline),
        ('estimator', search_cv.best_estimator_),
    ])
    EstimatorSerialize.save_estimator('svm', best_pipeline)

    print(search_cv.best_estimator_)
    print(search_cv.best_score_)


if __name__ == '__main__':
    main()
