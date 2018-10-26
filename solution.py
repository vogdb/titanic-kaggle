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
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from tabulate import tabulate


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._mean_mrs_age = np.floor(
            X.loc[X.Age.notnull() & X.Name.str.contains('Mrs')]['Age'].mean()
        )
        self._mean_age_greater_15 = np.floor(
            X.loc[(X.Age >= 15) & (X.Parch == 0) & (X.SibSp == 0)]['Age'].mean()
        )
        self._mean_age_male_3class = np.floor(
            X.loc[(X.Sex == 'male') & (X.Pclass == 3)]['Age'].mean()
        )
        self._mean_age_female_3class = np.floor(
            X.loc[(X.Sex == 'female') & (X.Pclass == 3)]['Age'].mean()
        )
        return self

    def transform(self, X, y=None):
        X.loc[X.Age.isnull() & X.Name.str.contains('Mrs'), 'Age'] = self._mean_mrs_age
        # there is 400 records of people with age >= 15 and 0 relatives vs 4 records of age < 15 and 0 relatives
        X.loc[X.Age.isnull() & (X.Parch == 0) & (X.SibSp == 0), 'Age'] = self._mean_age_greater_15
        # the remaining null Age contains only 3 class
        X.loc[X.Age.isnull() & (X.Sex == 'male') & (X.Pclass == 3), 'Age'] = self._mean_age_male_3class
        X.loc[X.Age.isnull() & (X.Sex == 'female') & (X.Pclass == 3), 'Age'] = self._mean_age_female_3class

        return X


class NewColumnsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['AgeBucket'] = X.Age // 15 * 15
        X['RelativesOnBoard'] = X.SibSp + X.Parch
        # is a reverend, they all die :(
        X['Rev'] = X.Name.str.contains('rev.|Rev.')
        return X


class ComparisonDiagramsEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_list=None, cv=10, save_path='./estimator_comparison', diagram_ext='png'):
        if estimator_list is None or not isinstance(estimator_list, (list, tuple)):
            raise ValueError('`estimator_list` argument should be a list of estimators')
        self.estimator_list = estimator_list
        self.save_path = save_path
        self.cv = cv
        self.diagram_ext = diagram_ext

        os.makedirs(save_path, exist_ok=True)

    def fit(self, X, y=None):
        self.boxplots(X, y)
        self.precision_recall(X, y)
        return self

    def boxplots(self, X, y):
        '''
        run cross_validation and display box plots (of cv's) of precision,recall,f1,accuracy for each model.
        '''
        score_name_list = ['accuracy', 'precision', 'recall', 'f1']
        crossval_data = []
        for clf in self.estimator_list:
            clf_name = type(clf).__name__
            score_dict = cross_validate(clf, X, y, scoring=score_name_list, cv=self.cv, return_train_score=False)
            for score_name in score_name_list:
                key = 'test_' + score_name
                for value in score_dict[key]:
                    crossval_data.append([clf_name, score_name, value])

        crossval_df = pd.DataFrame(data=crossval_data, columns=['clf_name', 'score_type', 'score_value'])
        print(
            tabulate(crossval_df, headers='keys', tablefmt='psql'),
            file=open(self.get_save_path('crossval_df.txt'), 'w')
        )
        crossval_dscr = crossval_df.groupby(['clf_name', 'score_type']).describe()
        print(
            tabulate(crossval_dscr, headers='keys', tablefmt='psql'),
            file=open(self.get_save_path('crossval_dscr.txt'), 'w')
        )

        boxplot = sns.boxplot(x='clf_name', y='score_value', hue='score_type', data=crossval_df)
        boxplot.set_ylim(0.5, 1.0)
        self.save_figure(boxplot, self.boxplots.__name__)
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

        self.save_figure(precision_vs_recall_ax, 'precision_vs_recall')
        self.save_figure(precision_recall_vs_threshold_ax, 'precision_recall_vs_threshold')
        self.save_figure(roc_curve_ax, 'roc_curve')

    def save_figure(self, ax, name):
        ax.figure.savefig(self.get_save_path(name + '.' + self.diagram_ext))

    def get_save_path(self, name):
        return os.path.join(self.save_path, name)


def create_prepare_pipeline():
    embarked_transformer = Pipeline([
        ('fill na', MostFrequentImputer()),
        ('onehot', OneHotEncoder(sparse=False)),
    ])

    column_transformer = ColumnTransformer([
        ('pass', 'passthrough', ['RelativesOnBoard', 'Rev']),
        ('scaler', StandardScaler(), ['Fare', 'Age']),
        ('cat', OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'AgeBucket']),
        ('embarked', embarked_transformer, ['Embarked']),
    ], remainder='drop')

    return Pipeline([
        ('age', AgeImputer()),
        ('new_columns', NewColumnsTransformer()),
        ('prepare', column_transformer),
    ])


class EstimatorSerialize:
    PATH = './saved_estimators'

    @staticmethod
    def check_dir():
        os.makedirs(EstimatorSerialize.PATH, exist_ok=True)

    @staticmethod
    def load_saved_estimators():
        EstimatorSerialize.check_dir()
        estimator_list = []
        for f in os.listdir(EstimatorSerialize.PATH):
            f_path = os.path.join(EstimatorSerialize.PATH, f)
            if os.path.isfile(f_path) and f.lower().endswith('.pkl'):
                estimator_list.append(joblib.load(f_path))
        return estimator_list

    @staticmethod
    def save_estimator(estimator):
        EstimatorSerialize.check_dir()
        joblib.dump(estimator, os.path.join(EstimatorSerialize.PATH, type(estimator).__name__ + '.pkl'))


def save_best_estimator(X, y):
    def save_svm():
        svm_clf = SVC()
        param_distribs = {
            'kernel': ['linear', 'rbf'],
            'C': reciprocal(20, 200000),
            'gamma': expon(scale=1.0),
        }
        search = RandomizedSearchCV(
            svm_clf, param_distributions=param_distribs,
            n_iter=40, cv=5, scoring='f1',
        )
        search.fit(X, y)
        EstimatorSerialize.save_estimator(search.best_estimator_)

    def save_rnd_forest():
        forest_clf = RandomForestClassifier()
        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        search = RandomizedSearchCV(
            forest_clf, param_distributions=param_distribs,
            n_iter=20, cv=5, scoring='f1',
        )
        search.fit(X, y)
        EstimatorSerialize.save_estimator(search.best_estimator_)

    save_svm()
    save_rnd_forest()


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
    main_pipeline.fit(df, df.Survived)
