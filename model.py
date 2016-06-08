from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

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
    categ_map = pd.read_csv('data/Category.csv', index_col='categoryID')
    categ = pairs.join(categ).merge(categ_map, how='left', left_on='categoryID', right_index=True)
    return categ[['parentCategoryID']].values


def basic_stats(data, labels, categ):
    print(pd.DataFrame(data).groupby(labels).describe())
    categ_df = pd.DataFrame(np.concatenate((categ, labels.reshape((-1, 1))), axis=1), columns=['categ', 'label'])
    print(categ_df.groupby('categ')['label'].agg(['mean', 'count']))


class MultEstimator(BaseEstimator):
    def __init__(self, categories):
        self.categories = categories

    def fit(self, X, y, **params):
        self.models = {_: None for _ in self.categories}
        self.tot_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1000)
        categ = X[:, -1]
        data = X[:, :-1]
        self.tot_model.fit(data, y)
        for c in self.models.keys():
            mask = categ == c
            m = GradientBoostingClassifier(min_samples_leaf=1000, max_depth=5, subsample=0.5, max_features=0.5)
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


class CombinedModel(BaseEstimator):
    def fit(self, X, y):
        # 0.791, 0.822, 0.782
        self.model1 = GradientBoostingClassifier(n_estimators=300, min_samples_leaf=1000, max_depth=5, subsample=0.5, max_features=1)
        self.model2 = GradientBoostingClassifier(n_estimators=300, min_samples_leaf=1000, max_depth=5, subsample=0.5, max_features=1)
        # self.model3 = DecisionTreeClassifier(min_samples_leaf=100, max_depth=8)
        self.model3 = GradientBoostingClassifier(n_estimators=300, min_samples_leaf=1000, max_depth=5, subsample=0.5, max_features=0.5)

        X1 = X[:, :6]
        # X3_1 = X[:, 6:8]
        X2 = X[:, 8:10]
        X3 = np.concatenate((X[:, 10:], X[:, 6:8]), axis=1)
        X1_, _, y1, _ = train_test_split(X1, y, test_size=0.5)  # topics
        X2_, _, y2, _ = train_test_split(X2, y, test_size=0.5)  # other
        X3_, _, y3, _ = train_test_split(X3, y, test_size=0.5)  # image sim
        # X4_, _, y4, _ = train_test_split(X4, y, test_size=0.5)  # categories

        self.model1.fit(X1_, y1)
        self.model2.fit(X2_, y2)
        self.model3.fit(X3_, y3)
        # self.model4.fit(X4_, y4)
        print(
            roc_auc_score(y, self.model1.predict_proba(X1)[:, 1]),
            roc_auc_score(y, self.model2.predict_proba(X2)[:, 1]),
            roc_auc_score(y, self.model3.predict_proba(X3)[:, 1]),
            # roc_auc_score(y, self.model4.predict_proba(X4)[:, 1])
        )
        self.model = GradientBoostingClassifier(n_estimators=300, min_samples_leaf=100, max_depth=8, subsample=0.5, max_features=1)
        self.model.fit(
            np.concatenate(
                (
                    self.model1.predict_proba(X1)[:, [1]],
                    self.model2.predict_proba(X2)[:, [1]],
                    self.model3.predict_proba(X3)[:, [1]],
                    # self.model4.predict_proba(X4)[:, [1]]
                ),
                axis=1),
            y
        )

    def predict_proba(self, X):
        X1 = X[:, :6]  # topics & attr_json
        X2 = X[:, 8:10]  # image
        X3 = np.concatenate((X[:, 10:], X[:, 6:8]), axis=1)  # price + categories
        return self.model.predict_proba(
            np.concatenate(
                (
                    self.model1.predict_proba(X1)[:, [1]],
                    self.model2.predict_proba(X2)[:, [1]],
                    self.model3.predict_proba(X3)[:, [1]],
                    # self.model4.predict_proba(X4)[:, [1]]
                ),
                axis=1)
        )

    @property
    def feature_importances_(self):
        print(
            self.model1.feature_importances_,
            self.model2.feature_importances_,
            self.model3.feature_importances_,
            # self.model4.feature_importances_
        )
        return self.model.feature_importances_ if hasattr(self.model, "feature_importances_")\
            else self.model._get_coef if hasattr(self.model, "_get_coef") else None

if __name__ == '__main__':
    s = datetime.now()
    data = load_data('train')
    print('data {}'.format(datetime.now() - s))
    labels = load_labels()
    print('labels {}'.format(datetime.now() - s))
    categ = load_category('train')
    print('category {}'.format(datetime.now() - s))
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
    # data = np.concatenate((data, categ), axis=1)

    # 0.93921317 combined - 0.92925244   0.924
    model = GradientBoostingClassifier(min_samples_leaf=100, max_depth=8, subsample=0.5, max_features=0.5)  # CombinedModel()
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    cross_val = cross_val_score(estimator=model, X=data, y=labels, scoring=scorer, verbose=2)
    print(cross_val)

    model.fit(data, labels)
    print(roc_auc_score(labels, model.predict_proba(data)[:, 1]))
    print(model.feature_importances_)
    test = load_data('test')
    categ = load_category('test')
    test_categ = varThr.transform(encoder.transform(categ))
    test[np.isnan(test)] = -1

    test = np.concatenate((test, test_categ.todense()), axis=1)
    # test = np.concatenate((test, categ), axis=1)
    pred = model.predict_proba(test)[:, 1]
    subm = pd.DataFrame(pred, columns=['probability'])
    subm.to_csv('submission.csv', index_label='id')
