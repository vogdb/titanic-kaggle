from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

from titanic_kaggle.data_processing import load_data
from titanic_kaggle.estimator_serialize import EstimatorSerialize


def main():
    rnd_forest_pipeline = EstimatorSerialize.load_estimator('rnd_forest')
    svm_pipeline = EstimatorSerialize.load_estimator('svm')
    mlp_pipeline = EstimatorSerialize.load_estimator('mlp')
    boost_tree_pipeline = EstimatorSerialize.load_estimator('boost_tree')

    estimator_list = [
        ('rnd_forest', rnd_forest_pipeline),
        ('svm', svm_pipeline),
        ('mlp', mlp_pipeline),
        ('boost_tree', boost_tree_pipeline),
    ]

    voting_clf_hard = VotingClassifier(
        estimators=estimator_list,
        voting='hard'
    )
    voting_clf_soft = VotingClassifier(
        estimators=estimator_list,
        voting='soft'
    )

    cmp_clf_list = estimator_list + [
        ('voting_hard', voting_clf_hard),
        ('voting_soft', voting_clf_soft),
    ]

    X_train, y_train, X_test = load_data()

    for clf_tuple in cmp_clf_list:
        clf_name, clf = clf_tuple
        # scores = cross_validate(
        #     clf, X_train, y_train, scoring=['accuracy', 'f1'], cv=3, n_jobs=-1, return_train_score=False,
        #     verbose=2
        # )
        # print(scores)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        print('{} acc: {}, f1: {}'.format(
            clf_name,
            accuracy_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred),
        ))

    y_test = boost_tree_pipeline.predict(X_test)
    X_test['Survived'] = y_test
    X_test.to_csv('result.csv', columns=['PassengerId', 'Survived'], header=True, index=False)


if __name__ == '__main__':
    main()
