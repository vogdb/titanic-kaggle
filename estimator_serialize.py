import os

from sklearn.externals import joblib


class EstimatorSerialize:
    PATH = './best_estimators'

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
    def load_estimator(name):
        estimator_path = os.path.join(EstimatorSerialize.PATH, name + '.pkl')
        return joblib.load(estimator_path)

    @staticmethod
    def save_estimator(name, estimator):
        EstimatorSerialize.check_dir()
        joblib.dump(estimator, os.path.join(EstimatorSerialize.PATH, name + '.pkl'))
