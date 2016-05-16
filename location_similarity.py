import time
from csv import DictReader, writer

import pandas as pd
import numpy as np


def location_similarity(lat_lon1, lat_lon2):
    diff = lat_lon1 - lat_lon2
    return np.linalg.norm(diff, 2)


if __name__ == "__main__":
    loc_and_category = pd.read_csv(
        'data/ItemInfo_train.csv', index_col='itemID',
        usecols=["itemID", "lat", 'lon', 'categoryID'], squeeze=True
    )
    print(loc_and_category.head(5))

    # calc similarities
    ts = time.time()
    misc_similarity = []
    with open('tmp/train_misc_similarity.csv', "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_train.csv') as f:

            dict_reader = DictReader(f)
            for i, row in enumerate(dict_reader):
                i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                misc_similarity.append(
                    [
                        location_similarity(
                            loc_and_category.loc[i1, ['lat', 'lon']].values,
                            loc_and_category.loc[i2, ['lat', 'lon']].values
                        ),
                        int(loc_and_category.loc[i1, 'categoryID'] == loc_and_category.loc[i2, 'categoryID'])
                    ]
                )
                if not i % 10000:
                    w.writerows(misc_similarity)
                    misc_similarity = []
                    print('{} {}'.format(i, time.time() - ts))

        w.writerows(misc_similarity)


