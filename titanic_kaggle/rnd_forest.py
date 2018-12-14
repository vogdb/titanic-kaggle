from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_kaggle.data_processing import load_data, MostFrequentImputer, NewColumnsTransformer, AgeImputer
from titanic_kaggle.estimator_serialize import EstimatorSerialize


def create_prepare_pipeline():
    embarked_transformer = Pipeline([
        ('fill na', MostFrequentImputer()),
        ('onehot', OneHotEncoder(sparse=False)),
    ])

    column_transformer = ColumnTransformer([
        ('pass', 'passthrough', ['RelativesOnBoard', 'Rev']),
        ('fare', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
        ]), ['Fare']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'AgeBucket']),
        ('embarked', embarked_transformer, ['Embarked']),
    ], remainder='drop')

    return Pipeline([
        ('age', AgeImputer()),
        ('new_columns', NewColumnsTransformer()),
        ('prepare', column_transformer),
    ])


def main():
    X_train, y_train, X_test = load_data()

    prepare_pipeline = create_prepare_pipeline()
    X_train = prepare_pipeline.fit_transform(X_train)
    forest_clf = RandomForestClassifier()
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
    search = RandomizedSearchCV(
        forest_clf, param_distributions=param_distribs,
        n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise'
    )
    search.fit(X_train, y_train)
    EstimatorSerialize.save_estimator('rnd_forest', search.best_estimator_)

    print(search.best_estimator_)
    print(search.best_score_)


if __name__ == '__main__':
    main()
