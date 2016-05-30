import time
from csv import DictReader, writer

import pandas as pd
import numpy as np

from text_to_similarity import load_tokens

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


def model_sim(m1, m2):
    # m1 = t1 # retrieve_model(t1)
    # m2 = t2 # retrieve_model(t2)
    return int((m1 in m2) or (m2 in m1))


def exact_sim(x1, x2):
    if not x1 or not x2:
        return 2
    elif x1 == x2:
        return 1
    else:
        return 0

if __name__ == "__main__":
    for prefix in ['train', 'test']:
        data = pd.read_csv(
            'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
            usecols=["itemID", "lat", 'lon', 'price'], squeeze=True
        )
        print(data.head(5))

        # calc similarities
        ts = time.time()
        misc_similarity = []
        with open('tmp/{}_misc_similarity.csv'.format(prefix), "w") as f_out:
            w = writer(f_out)
            with open('tmp/ItemPairs_{}.csv'.format(prefix)) as f:

                dict_reader = DictReader(f)
                for i, row in enumerate(dict_reader):
                    i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                    tl1, tl2 = len(row['t_text_1'].split()), len(row['t_text_2'].split())
                    dl1, dl2 = len(row['d_text_1'].split()), len(row['d_text_2'].split())
                    p1, p2 = data.loc[i1, 'price'], data.loc[i2, 'price']
                    misc_similarity.append(
                        [
                            location_similarity(data.loc[i1, ["lat", "lon"]], data.loc[i1, ["lat", "lon"]]),
                            min(dl1, dl2),
                            min(tl1, tl2),
                            model_sim(row['t_model_1'], row['t_model_2']),
                            min(p1 / p2, p2 / p1) if p1 and p2 else -1 if not (p1 or p2) else -2
                        ]
                    )
                    if not i % 100000:
                        w.writerows(misc_similarity)
                        misc_similarity = []
                        print('{} {}'.format(i, time.time() - ts))

            w.writerows(misc_similarity)


