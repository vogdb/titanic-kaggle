from scipy.stats import expon, reciprocal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from titanic_kaggle.data_processing import MostFrequentImputer, AgeImputer, NewColumnsTransformer, load_data
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


def main():
    X_train, y_train, X_test = load_data()
    prepare_pipeline = create_prepare_pipeline()
    X_train = prepare_pipeline.fit_transform(X_train)

    svm_clf = SVC()
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 20000),
        'gamma': expon(scale=1.0),
    }
    search = RandomizedSearchCV(
        svm_clf, param_distributions=param_distribs,
        n_iter=20, cv=5, scoring='accuracy', n_jobs=-1
    )
    search.fit(X_train, y_train)
    EstimatorSerialize.save_estimator('svm', search.best_estimator_)


if __name__ == '__main__':
    main()
