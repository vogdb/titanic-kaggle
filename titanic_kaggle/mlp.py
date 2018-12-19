import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from titanic_kaggle.data_processing import load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize
from titanic_kaggle.svm import create_prepare_pipeline


def rand_array():
    # return np.random.randint(100, size=(np.random.randint(1, 4)))
    return np.random.randint(50)


def rand_array_list(n):
    return [rand_array() for _ in range(n)]


def main():
    X_train, y_train, X_test = load_data()

    mlp_clf = MLPClassifier(
        [], solver='adam', batch_size=100, learning_rate='adaptive', alpha=.001, max_iter=1000,
        early_stopping=True, tol=.001, learning_rate_init=0.001, n_iter_no_change=100
    )
    param_grid = {
        'hidden_layer_sizes': rand_array_list(30),
    }
    search = GridSearchCV(
        mlp_clf, param_grid=param_grid, refit=True,
        cv=5, scoring='accuracy', n_jobs=-1, error_score='raise'
    )
    pipeline = Pipeline([
        ('prepare', create_prepare_pipeline()),
        ('search', search),
    ])

    pipeline.fit(X_train, y_train)
    EstimatorSerialize.save_estimator('mlp', pipeline)

    print(search.best_estimator_)
    print(search.best_score_)


if __name__ == '__main__':
    main()
