import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from titanic_kaggle.data_processing import load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize
from titanic_kaggle.rnd_forest import create_prepare_pipeline


def main():
    X_train, y_train, X_test = load_data()
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
    prepare_pipeline = create_prepare_pipeline()
    search_cv = GridSearchCV(
        clf, param_grid=param_grid, refit=True,
        cv=5, scoring='accuracy', n_jobs=-1, error_score='raise'
    )
    search_pipeline = Pipeline([
        ('prepare', prepare_pipeline),
        ('search', search_cv),
    ])
    search_pipeline.fit(X_train, y_train)

    best_pipeline = Pipeline([
        ('prepare', prepare_pipeline),
        ('estimator', search_cv.best_estimator_),
    ])
    EstimatorSerialize.save_estimator('boost_tree', best_pipeline)

    print(search_cv.best_estimator_)
    print(search_cv.best_score_)


if __name__ == '__main__':
    main()
