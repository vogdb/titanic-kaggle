import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import expon, reciprocal, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from tabulate import tabulate


# To the comparison and after saving the best

def write_result():
    df_train = pd.read_csv(os.path.join('dataset', 'train.csv'))
    df_test = pd.read_csv(os.path.join('dataset', 'test.csv'))
    prepare_pipeline = create_prepare_pipeline()
    estimator_class = RandomForestClassifier
    estimator = EstimatorSerialize.load_estimator(estimator_class)

    prepare_pipeline.fit(df_train, df_train.Survived)
    X_test = prepare_pipeline.transform(df_test)
    y_test = estimator.predict(X_test)
    df_test['Survived'] = y_test
    df_test.to_csv('result.csv', columns=['PassengerId', 'Survived'], header=True, index=False)


# TODO when we use the best SVM, we also need to use its prepare for transform of Test data.
calc_result = True
if calc_result:
    write_result()
else:
    df = pd.read_csv(os.path.join('dataset', 'train.csv'))
    y = df.Survived
    prepare_pipeline = create_prepare_pipeline()
    clf_list = EstimatorSerialize.load_saved_estimators()

    if len(clf_list) == 0:
        X = prepare_pipeline.fit_transform(df)
        save_best_estimator(X, y),
    else:
        main_pipeline = Pipeline([
            ('prepare', prepare_pipeline),
            ('analyse', ComparisonDiagramsEstimator(clf_list, cv=5)),
        ])
        main_pipeline.fit(df, y)
