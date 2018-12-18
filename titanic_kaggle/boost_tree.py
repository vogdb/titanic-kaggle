import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from titanic_kaggle.data_processing import load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize
from titanic_kaggle.rnd_forest import create_prepare_pipeline


def main():
    X_train, y_train, X_test = load_data()
    prepare_pipeline = create_prepare_pipeline()
    X_train = prepare_pipeline.fit_transform(X_train)
    use_xgb = True

    if use_xgb:
        clf = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1)
        param_grid = {
            'max_depth': [3],  # [2, 3, 4]
            'learning_rate': [.1],  # [.2 - .01]
            'n_estimators': [700],  # [50 - 7000]
        }
    else:
        clf = GradientBoostingClassifier()
        param_grid = {
            'max_depth': [3, 4],
            'learning_rate': [.2, .1],
            'n_estimators': [100, 400],
        }

    search = GridSearchCV(
        clf, param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    search.fit(X_train, y_train)
    EstimatorSerialize.save_estimator('boost_tree', search.best_estimator_)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.cv_results_['params'])
    print(search.cv_results_['mean_test_score'])


if __name__ == '__main__':
    main()
