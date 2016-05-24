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
    return np.linalg.norm(diff, 2)


def retrieve_model(t):
    model_descr = ""
    for token in t:
        token_ = re.sub(non_chars, "", token)
        # if re.search(only_chars, token_):
        model_descr += token_
    return model_descr.lower()


def model_sim(t1, t2):
    m1 = retrieve_model(t1)
    m2 = retrieve_model(t2)
    return int((m1 in m2) or (m2 in m1))


def exact_sim(x1, x2):
    if not x1 or not x2:
        return 2
    elif x1 == x2:
        return 1
    else:
        return 0

if __name__ == "__main__":
    prefix = 'train'
    # load description
    description = load_tokens('tmp/{}_description_tokens.csv'.format(prefix))
    print(description.head(5))

    # load title
    title = load_tokens('tmp/{}_title_tokens.csv'.format(prefix))
    print(title.head(5))

    data = pd.read_csv(
        'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
        usecols=["itemID", "lat", 'lon', 'categoryID', 'price'], squeeze=True
    )
    print(data.head(5))

    # calc similarities
    ts = time.time()
    misc_similarity = []
    with open('tmp/{}_misc_similarity.csv'.format(prefix), "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_{}.csv'.format(prefix)) as f:

            dict_reader = DictReader(f)
            for i, row in enumerate(dict_reader):
                i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                tl1, tl2 = len(title[i1] or []), len(title[i2] or [])
                dl1, dl2 = len(description[i1] or []), len(description[i2] or [])
                misc_similarity.append(
                    [
                        location_similarity(
                            data.loc[i1, ['lat', 'lon']].values,
                            data.loc[i2, ['lat', 'lon']].values
                        ),
                        exact_sim(data.loc[i1, 'categoryID'], data.loc[i2, 'categoryID']),
                        min(dl1, dl2),
                        min(tl1, tl2),
                        min(tl1/tl2, tl2/tl1) if tl1 and tl2 else -1,
                        min(dl1/dl2, dl2/dl1) if dl1 and dl2 else -1,
                        model_sim(title[i1], title[i2]),
                        exact_sim(title[i1], title[i2]),
                        exact_sim(description[i1], description[i2]),
                        min(data.loc[i1, 'price'] / data.loc[i2, 'price'], data.loc[i2, 'price'] / data.loc[i1, 'price'])\
                            if data.loc[i1, 'price'] and data.loc[i2, 'price'] else -1

                    ]
                )
                if not i % 10000:
                    w.writerows(misc_similarity)
                    misc_similarity = []
                    print('{} {}'.format(i, time.time() - ts))

        w.writerows(misc_similarity)


