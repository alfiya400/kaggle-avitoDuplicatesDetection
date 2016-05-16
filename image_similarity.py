import time
from csv import DictReader, writer, reader

from skimage.measure import structural_similarity as ssim
from skimage.io import imread

import pandas as pd


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
    archive = folder % 10
    return 'data/{archive}/{folder}/{im}.jpg'.format(
        archive=archive, folder=folder, im=im_id
    )


def image_similarity(im_id1, im_id2):
    """
    Calculates a structured similarity score for 2 images
    :param im_id1: image id from the avito competition
    :type im_id1: str
    :param im_id2: image id from the avito competition
    :type im_id2: str
    :return:
    """
    im_file1 = image_id_to_filepath(im_id1)
    im_file2 = image_id_to_filepath(im_id2)
    im1 = imread(im_file1, as_grey=True)
    im2 = imread(im_file2, as_grey=True)
    return ssim(im1, im2)


def get_max_similarity(im_array1, im_array2):
    """
    Finds the max similarity between 2 sets of images
    :param im_array1: string that contains image ids separated by ", "
    :type im_array1: str
    :param im_array2: string that contains image ids separated by ", "
    :type im_array2: str
    :return:
    """
    identical_score = 0.99  # used to break earlier
    im_id_list1 = im_array1.split(", ")
    im_id_list2 = im_array2.split(", ")

    if not im_id_list1 or not im_id_list2:
        return -1

    max_sim = -1
    for im_id1 in im_id_list1:
        for im_id2 in im_id_list2:
            sim = image_similarity(im_id1, im_id2)
            max_sim = max(max_sim, sim)
            if max_sim > identical_score:
                break
        if max_sim > identical_score:
            break

    return max_sim


if __name__ == "__main__":
    images_array = pd.read_csv(
        'data/ItemInfo_train.csv', index_col='itemID',
        usecols=["itemID", "images_array"], squeeze=True
    )
    # calc similarities
    ts = time.time()
    images_similarity = []
    with open('tmp/train_image_similarity.csv', "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_train.csv') as f:

            dict_reader = DictReader(f)
            for i, row in enumerate(dict_reader):
                i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                images_similarity.append(
                    get_max_similarity(images_array[i1], images_array[i2])
                )
                if not i % 10000:
                    w.writerows(images_similarity)
                    images_similarity = []
                    print('{} {}'.format(i, time.time() - ts))

        w.writerows(images_similarity)
