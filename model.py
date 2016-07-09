from datetime import datetime
from csv import reader
from keras.layers.core import Dropout
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

np.set_printoptions(suppress=True)


def load_image_sim(dataset='train'):
    res = []
    with open('tmp/{}_images_array_sims.csv'.format(dataset)) as f:
        r = reader(f)
        for row in r:
            if row[0] in ('-1', '-2'):
                res.append([int(row[0])]*2)
            else:
                i, j = int(row.pop(0)), int(row.pop(0))
                if i < j:
                    imsim = np.array(row, dtype=float) #.reshape((i, j)).max(axis=1)
                else:
                    imsim = np.array(row, dtype=float) #.reshape((i, j)).max(axis=0)
                s1 = imsim[imsim > 0.6].size
                s2 = imsim[imsim > 0.9].size
                res.append([float(s1) / (imsim.size - s1), float(s2) / (imsim.size - s2), min(i, j), min(float(i)/j, float(j)/i)])
    return np.array(res)


def load_data(dataset='train'):
    text = np.loadtxt('tmp/{}_text_similarity.csv'.format(dataset), delimiter=',')
    misc = np.loadtxt('tmp/{}_misc_similarity.csv'.format(dataset), delimiter=',')
    images = load_image_sim(dataset)
    return np.concatenate((text, misc, images), axis=1)


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


def load_region(prefix='train'):
    pairs = pd.read_csv('data/ItemPairs_{}.csv'.format(prefix), index_col='itemID_1', usecols=['itemID_1'], squeeze=True)
    categ = pd.read_csv(
        'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
        usecols=["itemID", 'locationID'], squeeze=True
    )
    categ_map = pd.read_csv('data/Location.csv', index_col='locationID')
    categ = pairs.join(categ).merge(categ_map, how='left', left_on='locationID', right_index=True)
    return categ[['regionID']].values


def basic_stats(data, labels, categ, region):
    print(pd.DataFrame(data).groupby(labels).describe())
    categ_df = pd.DataFrame(np.concatenate((categ, labels.reshape((-1, 1))), axis=1), columns=['categ', 'label'])
    print(categ_df.groupby('categ')['label'].agg(['mean', 'count']))
    df = pd.DataFrame(np.concatenate((region, labels.reshape((-1, 1))), axis=1), columns=['c', 'l']).groupby('c')['l'].agg(['mean', 'count'])
    keep = (((df['mean'] <= 0.4) | (df['mean'] >= 0.55)) & (df['count'] > 10000)).values
    print(df[keep])


class MultEstimator(BaseEstimator):
    def __init__(self, categories):
        self.categories = categories

    def fit(self, X, y, **params):
        self.models = {_: None for _ in self.categories}
        self.tot_model = XGBClassifier(min_child_weight=100, max_depth=8, subsample=0.5, colsample_bytree=0.5)
        categ = X[:, -1]
        data = X[:, :-1]
        self.tot_model.fit(data, y)
        for c in self.models.keys():
            mask = categ == c
            m = XGBClassifier(n_estimators=200, min_child_weight=100, max_depth=8, subsample=0.5, colsample_bytree=0.5)
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


class toDummies(BaseEstimator, TransformerMixin):
    def __init__(self, var_threshold=0.02):
        self.var_threshold = var_threshold

    def fit(self, X, y=None, **kwargs):
        X = X.copy()
        if y is not None:
            df = pd.DataFrame(np.concatenate((X, y.reshape((-1, 1))), axis=1), columns=['c', 'l']).groupby('c')['l'].mean()
            m1, m2 = 0.4, 0.55
            ignore = df[(df <= m2) & (df >= m1)].index.values
            mask = np.in1d(X.ravel(), ignore)
            X[mask] = 0

        self.encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float16)
        dummies = self.encoder.fit_transform(X)
        print(dummies.shape)
        self.varThr = VarianceThreshold(self.var_threshold)
        self.varThr.fit(dummies)
        return self

    def transform(self, X, y=None, **kwargs):
        cut_dummies = self.varThr.fit_transform(self.encoder.transform(X))
        print(cut_dummies.shape)
        return cut_dummies


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
    region = load_region('train')
    print('region {}'.format(datetime.now() - s))
    basic_stats(data, labels, categ, region)

    data[np.isnan(data)] = -1
    categ_transformer = toDummies(0.02)
    categ_transformer.fit(categ)

    region_transformer = toDummies(0.01)
    region_transformer.fit(region)
    data = np.concatenate((
        data,
        categ_transformer.transform(categ).todense(),
        region_transformer.transform(region).todense()
    ), axis=1)
    print(data.shape)
    # data = np.concatenate((data, categ), axis=1)

    # 0.94025061
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=labels.mean(),
        gamma=1.)  # CombinedModel()
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    cross_val = cross_val_score(estimator=model, X=data, y=labels, scoring=scorer, verbose=2, n_jobs=3)
    print(cross_val)

    model.fit(data, labels)
    print('total train score', roc_auc_score(labels, model.predict_proba(data)[:, 1]))
    for c in np.unique(categ):
        is_c = categ.ravel() == c
        p = model.predict_proba(data[is_c])[:, 1]
        print(c, roc_auc_score(labels[is_c], p))
        if c == 110:
            mis_class = labels[is_c] != (p > 0.5)
            print(pd.Series(p).describe())
            # print(pd.read_csv('tmp/ItemPairs_train.csv')[is_c][mis_class])

    print(model.feature_importances_)
    test = load_data('test')
    categ = load_category('test')
    region = load_region('test')
    test[np.isnan(test)] = -1

    test = np.concatenate((
        test,
        categ_transformer.transform(categ).todense(),
        region_transformer.transform(region).todense()
    ), axis=1)
    # test = np.concatenate((test, categ), axis=1)
    pred = model.predict_proba(test)[:, 1]
    subm = pd.DataFrame(pred, columns=['probability'])
    subm.to_csv('submission.csv', index_label='id')
