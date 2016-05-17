import pandas as pd
import numpy as np


def load_data():
    text = np.loadtxt('tmp/train_text_similarity.csv', delimiter=',')
    misc = np.loadtxt('tmp/train_misc_similarity.csv', delimiter=',')
    # images = np.loadtxt('tmp/train_image_similarity.csv', delimiter=',')
    return np.concatenate((text, misc), axis=1)


def load_labels():
    pairs = pd.read_csv('data/ItemPairs_train.csv')
    return pairs['isDuplicate'].values


def basic_stats(data, labels):
    print(pd.DataFrame(data).groupby(labels).describe())

