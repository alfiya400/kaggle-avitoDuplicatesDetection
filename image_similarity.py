import time
from csv import DictReader, writer, reader

from skimage.measure import compare_ssim as ssim
from skimage import transform, util, img_as_uint
from skimage.io import imread
from skimage.restoration import denoise_nl_means

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
        im = img_as_uint(imread(im_file1, as_grey=True))
    except:
        im = None
    finally:
        return im


def image_similarity(im1, im2):
    """
    Calculates a structured similarity score for 2 images
    :param im1: image id from the avito competition
    :type im1: np.array
    :param im2: image id from the avito competition
    :type im2: np.array
    :return:
    """
    if min(im1.shape) < 15 or min(im2.shape) < 15:
        return 0

    shape = (min(im1.shape[0], im2.shape[0]), min(im1.shape[1], im2.shape[1]))
    # print(im1.shape, im2.shape, shape)
    x2, y2 = shape
    if im1.shape != shape:
        x, y = im1.shape
        x_d, y_d = (x - x2) / 2, (y - y2) / 2
        im1 = im1[x_d:x2 + x_d, y_d: y2 + y_d]
    if im2.shape != shape:
        x, y = im2.shape
        x_d, y_d = (x - x2) / 2, (y - y2) / 2
        im2 = im2[x_d:x2 + x_d, y_d: y2 + y_d]
    # print(im1.shape, im2.shape)
    if (im1 == im2).all():
        return 1.
    sim = ssim(im1, im2, gaussian_weights=True, use_sample_covariance=False)
    return round(sim, 2)


def get_similarities(im_array1, im_array2):
    """
    Finds the max similarity between 2 sets of images
    :param im_array1: string that contains image ids separated by ", "
    :type im_array1: str
    :param im_array2: string that contains image ids separated by ", "
    :type im_array2: str
    :return:
    """

    if not im_array1 and not im_array2:
        return [-2]
    if not im_array1 or not im_array2:
        return [-1]
    identical_score = 0.99  # used to break earlier
    im_id_list1 = im_array1.split(", ")  # if isinstance(im_array1, str) else [str(im_array1)]
    im_id_list2 = im_array2.split(", ")  # if isinstance(im_array2, str) else [str(im_array2)]

    im_list1 = [load_image(im_id) for im_id in im_id_list1]
    im_list2 = [load_image(im_id) for im_id in im_id_list2]
    im_list1 = [im for im in im_list1 if im is not None]
    im_list2 = [im for im in im_list2 if im is not None]
    if not im_list1 and not im_list2:
        return [-2]
    elif not im_list1 or not im_list2:
        return [-1]

    sim = [len(im_list1), len(im_list2)]
    for im1 in im_list1:
        for im2 in im_list2:
            sim.append(image_similarity(im1, im2))

    return sim


def sc(row):
    i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
    return get_similarities(images_array[i1], images_array[i2])

if __name__ == "__main__":
    prefix = 'test'
    images_array = pd.read_csv(
        'data/ItemInfo_{}.csv'.format(prefix), index_col='itemID',
        usecols=["itemID", "images_array"], squeeze=True
    )
    images_array = images_array.fillna("").astype(str)
    # calc similarities
    ts = time.time()

    with open('tmp/{}_images_array_sims.csv'.format(prefix), "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_{}.csv'.format(prefix)) as f:
            dict_reader = DictReader(f)
            data = []
            for i, row in enumerate(dict_reader):
                data.append(row)
                if not i % 10000 and i > 0:
                    images_similarity = Parallel(n_jobs=5)(delayed(sc)(r) for r in data)
                    # print(images_similarity)
                    w.writerows(images_similarity)
                    data = []
                    print('{} {}'.format(i, time.time() - ts))

            images_similarity = Parallel(n_jobs=5)(delayed(sc)(r) for r in data)
            # print(images_similarity)
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
