import os

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv(os.path.join('dataset', 'train.csv'))
y = df.Survived

forest_clf = RandomForestClassifier()
param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}
search = RandomizedSearchCV(
    forest_clf, param_distributions=param_distribs,
    n_iter=20, cv=5, scoring='accuracy', n_jobs=4
)
search.fit(X, y)
EstimatorSerialize.save_estimator('rnd_forest', search.best_estimator_)
