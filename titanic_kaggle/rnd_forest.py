from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_kaggle.data_processing import load_data, MostFrequentImputer, AgeImputer, NewColumnsTransformer
from titanic_kaggle.estimator_serialize import EstimatorSerialize


def create_prepare_pipeline():
    column_transformer = ColumnTransformer([
        ('pass', 'passthrough', ['RelativesOnBoard', 'Rev']),
        ('fare', SimpleImputer(strategy='median'), ['Fare']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'AgeBucket']),
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

    param_distribs = {
        'n_estimators': range(150, 170, 2),
        'max_features': range(10, 16),
    }
    search_cv = GridSearchCV(
        RandomForestClassifier(), param_grid=param_distribs, refit=True,
        cv=5, scoring='accuracy', n_jobs=-1, error_score='raise'
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
    EstimatorSerialize.save_estimator('rnd_forest', best_pipeline)

    print(search_cv.best_estimator_)
    print(search_cv.best_score_)


if __name__ == '__main__':
    main()
