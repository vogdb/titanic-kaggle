import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from titanic_kaggle.data_processing import load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize
from titanic_kaggle.svm import create_prepare_pipeline


def rand_array():
    return np.random.randint(100, size=(np.random.randint(1, 4)))


def rand_array_list(n):
    return [rand_array() for _ in range(n)]


def main():
    X_train, y_train, X_test = load_data()
    prepare_pipeline = create_prepare_pipeline()
    X_train = prepare_pipeline.fit_transform(X_train)
    mlp_clf = MLPClassifier(
        [], solver='adam', batch_size=100, learning_rate='adaptive', alpha=.001, max_iter=1000,
        early_stopping=True, tol=.001, learning_rate_init=0.001, n_iter_no_change=100
    )

    param_grid = {
        'hidden_layer_sizes': rand_array_list(30),
    }
    search = GridSearchCV(
        mlp_clf, param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    search.fit(X_train, y_train)
    EstimatorSerialize.save_estimator('mlp', search.best_estimator_)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.cv_results_['mean_test_score'])


if __name__ == '__main__':
    main()
