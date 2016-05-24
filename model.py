import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer


def load_data(dataset='train'):
    text = np.loadtxt('tmp/{}_text_similarity.csv'.format(dataset), delimiter=',')
    misc = np.loadtxt('tmp/{}_misc_similarity.csv'.format(dataset), delimiter=',')
    # images = np.loadtxt('tmp/train_image_similarity.csv', delimiter=',')
    return np.concatenate((text, misc), axis=1)


def load_labels():
    pairs = pd.read_csv('data/ItemPairs_train.csv')
    return pairs['isDuplicate'].values


def basic_stats(data, labels):
    print(pd.DataFrame(data).groupby(labels).describe())

if __name__ == '__main__':
    data = load_data('train')
    labels = load_labels()
    basic_stats(data, labels)
    print(data.shape, labels.shape)
    print(np.isnan(data).sum(axis=0))
    data[np.isnan(data)] = -1
    print(np.isnan(data).sum(axis=0))

    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=100)  # GradientBoostingClassifier(min_samples_leaf=100, max_depth=8)
    scorer = make_scorer(roc_auc_score, needs_threshold=True)
    cross_val = cross_val_score(estimator=model, X=data, y=labels, scoring=scorer, verbose=1)
    print(cross_val)

    model.fit(data, labels)
    print(roc_auc_score(labels, model.predict_proba(data)[:, 1]))
    test = load_data('test')
    test[np.isnan(test)] = -1
    pred = model.predict_proba(test)[:, 1]
    subm = pd.DataFrame(pred, columns=['probability'])
    subm.to_csv('submission.csv', index_label='id')
