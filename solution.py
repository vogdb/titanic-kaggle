import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


class ComparisonDiagramsEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_list=None, cv=10, diagram_save_path='./estimator_comparison', diagram_ext='png'):
        if estimator_list is None or not isinstance(estimator_list, (list, tuple)):
            raise ValueError('`estimator_list` argument should be a list of estimators')
        self.estimator_list = estimator_list
        self.diagram_save_path = diagram_save_path
        self.cv = cv
        self.diagram_ext = diagram_ext

        os.makedirs(diagram_save_path, exist_ok=True)

    def fit(self, X, y=None):
        self.boxplots(X, y)
        self.precision_recall(X, y)
        return self

    def boxplots(self, X, y):
        '''
        run cross_validation and display box plots (of cv's) of precision,recall,f1,accuracy for each model.
        '''
        score_name_list = ['accuracy', 'precision', 'recall', 'f1']
        boxplot_data = []
        for clf in self.estimator_list:
            clf_name = type(clf).__name__
            score_dict = cross_validate(clf, X, y, scoring=score_name_list, cv=self.cv, return_train_score=False)
            for score_name in score_name_list:
                key = 'test_' + score_name
                for value in score_dict[key]:
                    boxplot_data.append([clf_name, score_name, value])
        boxplot_df = pd.DataFrame(data=boxplot_data, columns=['clf_name', 'score_type', 'score_value'])
        boxplot = sns.boxplot(x='clf_name', y='score_value', hue='score_type', data=boxplot_df)
        filename = self.boxplots.__name__ + '.' + self.diagram_ext
        boxplot.figure.savefig(os.path.join(self.diagram_save_path, filename))
        plt.close()

    def precision_recall(self, X, y):
        '''
        display precision vs recall
        display precision, recall vs threshold
        display ROC curve
        '''

        def get_scores(clf, X, y, cv=self.cv):
            if hasattr(clf, 'decision_function'):
                return cross_val_predict(clf, X, y, cv=cv, method='decision_function')
            if hasattr(clf, 'predict_proba'):
                y_probas = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
                # take probabilities of positive class as scores
                return y_probas[:, 1]
            raise AttributeError('Estimator {} should have `decision_function` or `predict_proba`'.format(clf))

        def get_precision_vs_recall_ax():
            ax = plt.figure(1).gca()
            ax.set_title('Precision vs Recall')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.axis([0, 1, 0, 1])
            return ax

        def get_precision_recall_vs_threshold_ax():
            ax = plt.figure(2).gca()
            ax.set_title('Precision, Recall vs threshold')
            ax.set_ylabel('Precision, solid')
            ax_copy = ax.twinx()
            ax_copy.set_ylabel('Recall, dashed')
            ax.set_xlabel('Threshold')
            ax.set_ylim([0, 1])
            return ax

        def get_roc_curve_ax():
            ax = plt.figure(3).gca()
            ax.set_title('ROC curve')
            ax.axis([0, 1, 0, 1])
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            return ax

        basic_color_list = list(map(lambda color: color[1], matplotlib.colors.TABLEAU_COLORS.items()))

        precision_vs_recall_ax = get_precision_vs_recall_ax()
        roc_curve_ax = get_roc_curve_ax()
        precision_recall_vs_threshold_ax = get_precision_recall_vs_threshold_ax()

        for estimator in self.estimator_list:
            clf_name = type(estimator).__name__
            clf_color = np.random.choice(basic_color_list, replace=False)
            y_scores = get_scores(estimator, X, y)

            precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
            precision_vs_recall_ax.plot(recalls, precisions, clf_color, label=clf_name)
            precision_recall_vs_threshold_ax.plot(thresholds, precisions[:-1], clf_color, label=clf_name)
            precision_recall_vs_threshold_ax.plot(thresholds, recalls[:-1], '--', color=clf_color)

            fpr, tpr, _ = roc_curve(y, y_scores)
            roc_curve_ax.plot(fpr, tpr, clf_color, label=clf_name)

        precision_vs_recall_ax.legend()
        precision_recall_vs_threshold_ax.legend()
        roc_curve_ax.legend()

        precision_vs_recall_ax.figure.savefig(
            os.path.join(self.diagram_save_path, 'precision_vs_recall.' + self.diagram_ext)
        )
        precision_recall_vs_threshold_ax.figure.savefig(
            os.path.join(self.diagram_save_path, 'precision_recall_vs_threshold.' + self.diagram_ext)
        )
        roc_curve_ax.figure.savefig(
            os.path.join(self.diagram_save_path, 'roc_curve.' + self.diagram_ext)
        )


def create_prepare_pipeline():
    age_transformer = Pipeline([
        ('fill na', SimpleImputer(strategy='median')),
        # ('scale', StandardScaler()),
    ])

    embarked_transformer = Pipeline([
        ('fill na', MostFrequentImputer()),
        ('onehot', OneHotEncoder(sparse=False)),
    ])

    return ColumnTransformer([
        ('age', age_transformer, ['Age']),
        ('pass', 'passthrough', ['SibSp', 'Parch', 'Fare']),
        # ('num', StandardScaler(), ['SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex']),
        ('embarked', embarked_transformer, ['Embarked']),
    ], remainder='drop')


def create_analyse_pipeline():
    svm_clf = SVC(gamma='auto')
    forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
    clf_list = [svm_clf, forest_clf]
    return ComparisonDiagramsEstimator(clf_list, cv=5)


df = pd.read_csv(os.path.join('dataset', 'train.csv'))

prepare_pipeline = create_prepare_pipeline()
analyse_pipeline = create_analyse_pipeline()
main_pipeline = Pipeline([
    ('prepare', prepare_pipeline),
    ('analyse', analyse_pipeline),
])

main_pipeline.fit(df, df.Survived)