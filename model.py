import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from collections import defaultdict
np.set_printoptions(suppress=True)


def load_data(dataset='train'):
    text = np.loadtxt('tmp/{}_text_similarity.csv'.format(dataset), delimiter=',')
    misc = np.loadtxt('tmp/{}_misc_similarity.csv'.format(dataset), delimiter=',')
    images = np.loadtxt('tmp/{}_image_similarity.csv'.format(dataset), delimiter=',')
    return np.concatenate((text, misc, images.reshape((-1, 1))), axis=1)


def load_labels():
    pairs = pd.read_csv('data/ItemPairs_train.csv')
    return pairs['isDuplicate'].values


def load_category(prefix='train'):
    pairs = pd.read_csv('data/ItemPairs_{}.csv'.format(prefix), index_col='itemID_1', usecols=['itemID_1'], squeeze=True)
    categ = pd.read_csv(
        'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
        usecols=["itemID",'categoryID'], squeeze=True
    )
    return pairs.join(categ).values


def basic_stats(data, labels, categ):
    print(pd.DataFrame(data).groupby(labels).describe())
    # categ_df = pd.DataFrame(np.concatenate((categ, labels.reshape((-1, 1))), axis=1), columns=['categ', 'label'])
    # print(categ_df.groupby('categ')['label'].sum())


class MultEstimator(BaseEstimator):
    def __init__(self, categories):
        self.categories = categories

    def fit(self, X, y, **params):
        self.models = {_: None for _ in self.categories}
        self.tot_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=100)
        categ = X[:, -1]
        data = X[:, :-1]
        self.tot_model.fit(data, y)
        for c in self.models.keys():
            mask = categ == c
            m = DecisionTreeClassifier(max_depth=8, min_samples_leaf=100)
            m.fit(data[mask], y[mask])
            self.models[c] = m

    def predict(self, X):
        categ = X[:, -1]
        data = X[:, :-1]
        p = self.tot_model.predict(data)
        for c in self.models.keys():
            mask = categ == c
            if mask.any():
                p[mask] = self.models[c].predict(data[mask])
        return p

    def predict_proba(self, X):
        categ = X[:, -1]
        data = X[:, :-1]
        p = self.tot_model.predict_proba(data)
        for c in self.models.keys():
            mask = categ == c
            if mask.any():
                p[mask] = self.models[c].predict_proba(data[mask])
        return p

if __name__ == '__main__':
    data = load_data('train')
    labels = load_labels()
    categ = load_category('train')
    basic_stats(data, labels, categ)

    data[np.isnan(data)] = -1
    encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float16)
    categ_sparse = encoder.fit_transform(categ)
    varThr = VarianceThreshold(0.02)  # p * (1 - p), p = 10k / <train_nrows>
    print(categ_sparse.shape)
    train_categ = varThr.fit_transform(categ_sparse)
    print(train_categ.shape)
    print(data.shape)
    data = np.concatenate((data, train_categ.todense()), axis=1)

    model = GradientBoostingClassifier(min_samples_leaf=100, max_depth=8, subsample=0.5, max_features=0.5)
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    cross_val = cross_val_score(estimator=model, X=data, y=labels, scoring=scorer, verbose=2, n_jobs=3)
    print(cross_val)

    model.fit(data, labels)
    print(roc_auc_score(labels, model.predict_proba(data)[:, 1]))
    print(model.feature_importances_)
    test = load_data('test')
    test_categ = varThr.transform(encoder.transform(load_category('test')))
    test[np.isnan(test)] = -1

    test = np.concatenate((test, test_categ.todense()), axis=1)

    pred = model.predict_proba(test)[:, 1]
    subm = pd.DataFrame(pred, columns=['probability'])
    subm.to_csv('submission.csv', index_label='id')
