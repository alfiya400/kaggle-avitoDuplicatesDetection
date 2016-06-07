import time
from csv import DictReader, writer
from json import loads

import pandas as pd
import numpy as np

import re

non_chars = re.compile('\W+')
only_chars = re.compile('[A-Za-z]')


def location_similarity(lat_lon1, lat_lon2):
    diff = lat_lon1 - lat_lon2
    return np.linalg.norm(diff, 1)


def retrieve_model(t):
    model_descr = ""
    for token in t:
        token_ = re.sub(non_chars, "", token)
        # if re.search(only_chars, token_):
        model_descr += token_
    return model_descr.lower()


def model_sim(x):
    return int((x[0] in x[1]) or (x[1] in x[0]))


def exact_sim(x1, x2):
    if not x1 or not x2:
        return 2
    elif x1 == x2:
        return 1
    else:
        return 0


def json_sim(x):
    x1 = set(loads(x[0]).items())
    x2 = set(loads(x[1]).items())
    return float(len(x1.intersection(x2))) / min(len(x1), len(x2))


if __name__ == "__main__":
    for prefix in ['train', 'test']:
        data = pd.read_csv(
            'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
            usecols=["itemID", "lat", 'lon', 'price', 'images_array', 'attrsJSON']
        )
        print(data.head(5))

        # calc similarities
        ts = time.time()
        misc_similarity = []
        with open('tmp/{}_misc_similarity.csv'.format(prefix), "w") as f_out:
            w = writer(f_out)
            pairs = pd.read_csv('tmp/ItemPairs_{}.csv'.format(prefix), chunksize=100000)
            for i, chunk in enumerate(pairs):
                chunk = chunk.merge(
                    data, how='left', left_on='itemID_1', right_index=True
                ).merge(
                    data, how='left', left_on='itemID_2', right_index=True, suffixes=('_1', '_2')
                )
                nulls = pd.isnull(chunk)

                min_t_len = np.zeros((chunk.shape[0],))
                min_t_len[nulls[['t_text_1', 't_text_2']].values.any(axis=1)] = -1
                min_t_len[nulls[['t_text_1', 't_text_2']].values.all(axis=1)] = -2
                not_null = ~nulls[['t_text_1', 't_text_2']].values.any(axis=1)
                min_t_len[not_null] = np.minimum(
                    chunk.loc[not_null, 't_text_1'].str.split().apply(lambda x: len(x)).values,
                    chunk.loc[not_null, 't_text_2'].str.split().apply(lambda x: len(x)).values
                )

                min_d_len = np.zeros((chunk.shape[0],))
                min_d_len[nulls[['d_text_1', 'd_text_2']].values.any(axis=1)] = -1
                min_d_len[nulls[['d_text_1', 'd_text_2']].values.all(axis=1)] = -2
                not_null = ~nulls[['d_text_1', 'd_text_2']].values.any(axis=1)
                min_d_len[not_null] = np.minimum(
                    chunk.loc[not_null, 'd_text_1'].str.split().apply(lambda x: len(x or [])).values,
                    chunk.loc[not_null, 'd_text_2'].str.split().apply(lambda x: len(x or [])).values
                )

                attr_sim = np.zeros((chunk.shape[0],))
                attr_sim[nulls[['attrsJSON_1', 'attrsJSON_2']].values.any(axis=1)] = -1
                attr_sim[nulls[['attrsJSON_1', 'attrsJSON_2']].values.all(axis=1)] = -2
                not_null = ~nulls[['attrsJSON_1', 'attrsJSON_2']].values.any(axis=1)
                attr_sim[not_null] = chunk.loc[not_null, ['attrsJSON_1', 'attrsJSON_2']].apply(json_sim, raw=True, axis=1).values

                price_min = np.zeros((chunk.shape[0],))
                price_ratio = np.zeros((chunk.shape[0],))
                price_ratio[nulls[['price_1', 'price_2']].values.any(axis=1)] = -1
                price_ratio[nulls[['price_1', 'price_2']].values.all(axis=1)] = -2
                price_min = price_ratio.copy()
                not_null = ~nulls[['price_1', 'price_2']].values.any(axis=1)
                price_ratio[not_null] =\
                    np.minimum(chunk.loc[not_null, 'price_1'].values, chunk.loc[not_null, "price_2"].values) / \
                    (np.maximum(chunk.loc[not_null, 'price_2'].values, chunk.loc[not_null, "price_1"].values) + 1)

                price_min[not_null] = np.minimum(chunk.loc[not_null, 'price_1'].values, chunk.loc[not_null, "price_2"].values)

                m_sim = np.zeros((chunk.shape[0],))
                m_sim[nulls[['t_model_1', 't_model_2']].values.any(axis=1)] = -1
                m_sim[nulls[['t_model_1', 't_model_2']].values.all(axis=1)] = -2
                not_null = ~nulls[['t_model_1', 't_model_2']].values.any(axis=1)
                m_sim[not_null] =\
                    chunk.loc[not_null, ['t_model_1', 't_model_2']].apply(model_sim, axis=1, raw=True).values

                i_sim = np.zeros((chunk.shape[0],))
                i_sim[nulls[['images_array_1', 'images_array_2']].values.any(axis=1)] = -1
                i_sim[nulls[['images_array_1', 'images_array_2']].values.all(axis=1)] = -2

                misc_similarity = np.concatenate(
                    (
                        min_d_len.reshape((-1, 1)),
                        min_t_len.reshape((-1, 1)),
                        m_sim.reshape((-1, 1)),
                        attr_sim.reshape((-1, 1)),
                        price_ratio.reshape((-1, 1)),
                        price_min.reshape((-1, 1)),
                        i_sim.reshape((-1, 1))
                    ),
                    axis=1
                ).tolist()
                w.writerows(misc_similarity)
                print('{} {}'.format(i, time.time() - ts))


