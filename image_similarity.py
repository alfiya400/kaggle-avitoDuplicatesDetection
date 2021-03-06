import time
from csv import DictReader, writer, reader

from skimage.measure import structural_similarity as ssim
from skimage import transform, util
from skimage.io import imread
from skimage import filters

import pandas as pd
from joblib import Parallel, delayed

def image_id_to_filepath(im_id):
    """
    From kaggle:
    >   To find and image folder you need to take last 2 digits of image id and remove leading zero if any.
        Folders are grouped into archives by their first digit (0 if no digit).
        Example: you need to find image for image_id = 123456.
        Take last 2 digits = 56, thats why it is contained in 56 folder.
        This folder is contained in Images_5.zip archive.
    :param im_id: id of an image
    :type im_id: str
    :return:
    """
    folder = int(im_id[-2:])
    archive = 'Images_{}'.format(folder // 10)
    return 'data/{archive}/{folder}/{im}.jpg'.format(
        archive=archive, folder=folder, im=im_id
    )


def load_image(im_id):
    im = None
    try:
        im_file1 = image_id_to_filepath(im_id)
        im = imread(im_file1, as_grey=True).astype(float)
    except:
        im = None
    finally:
        return im


def image_similarity(im1, im2):
    """
    Calculates a structured similarity score for 2 images
    :param im_id1: image id from the avito competition
    :type im_id1: str
    :param im_id2: image id from the avito competition
    :type im_id2: str
    :return:
    """
    # eimg1 = filters.sobel(im1)
    # eimg2 = filters.sobel(im2)

    if im1.shape != (105, 140):
        im1 = transform.resize(im1, (105, 140))
    if im2.shape != (105, 140):
        im2 = transform.resize(im2, (105, 140))
    # print(im1.shape, im2.shape, im_file1, im_file2, im_id1, im_id2)
    sim = ssim(im1, im2)
    return sim


def get_max_similarity(im_array1, im_array2):
    """
    Finds the max similarity between 2 sets of images
    :param im_array1: string that contains image ids separated by ", "
    :type im_array1: str
    :param im_array2: string that contains image ids separated by ", "
    :type im_array2: str
    :return:
    """

    if not im_array1 or not im_array2:
        return -1
    identical_score = 0.99  # used to break earlier
    im_id_list1 = im_array1.split(", ")  # if isinstance(im_array1, str) else [str(im_array1)]
    im_id_list2 = im_array2.split(", ")  # if isinstance(im_array2, str) else [str(im_array2)]

    im_list1 = [load_image(im_id) for im_id in im_id_list1]
    im_list2 = [load_image(im_id) for im_id in im_id_list2]
    im_list1 = [im for im in im_list1 if im is not None]
    im_list2 = [im for im in im_list2 if im is not None]
    if not im_list1 or not im_list2:
        return -1

    max_sim = -1
    for im1 in im_list1:
        for im2 in im_list2:
            if im1.shape == im2.shape and (im1 == im2).all():
                max_sim = 1
                break
        if max_sim > identical_score:
            break

    if max_sim < identical_score:
        max_sim = -1
        for im_id1 in im_list1:
            for im_id2 in im_list2:
                sim = image_similarity(im_id1, im_id2)
                max_sim = max(max_sim, sim)
                if max_sim > identical_score:
                    break
            if max_sim > identical_score:
                break

    return max_sim


def sc(row):
    i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
    return [get_max_similarity(images_array[i1], images_array[i2])]

if __name__ == "__main__":
    prefix = 'test'
    images_array = pd.read_csv(
        'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
        usecols=["itemID", "images_array"], squeeze=True
    )
    images_array = images_array.fillna("").astype(str)
    print images_array.dtype
    # calc similarities
    ts = time.time()

    with open('tmp/{}_image_similarity.csv'.format(prefix), "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_{}.csv'.format(prefix)) as f:
            dict_reader = DictReader(f)
            data = []
            for i, row in enumerate(dict_reader):
                data.append(row)
                if not i % 10000 and i > 0:
                    images_similarity = Parallel(n_jobs=5)(delayed(sc)(r) for r in data)
                    w.writerows(images_similarity)
                    data = []
                    print('{} {}'.format(i, time.time() - ts))

            images_similarity = Parallel(n_jobs=5)(delayed(sc)(r) for r in data)
            w.writerows(images_similarity)
            print('last batch {} {}'.format(i, time.time() - ts))
            # for i, row in enumerate(dict_reader):
            #     i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
            #     images_similarity.append(
            #         [get_max_similarity(images_array[i1], images_array[i2])]
            #     )
            #     if not i % 100 and i > 0:
            #         w.writerows(images_similarity)
            #         images_similarity = []
            #         print('{} {}'.format(i, time.time() - ts))
