# coding=utf-8
from csv import DictReader, writer, reader
from itertools import chain, islice
from os.path import basename, splitext
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer

import re

non_chars = re.compile('\W+')
non_russian = re.compile(unicode('[^йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ]', 'utf-8'))
# non_russian = re.compile(u'[^\u0439\u0446\u0443\u043a\u0435\u043d\u0433\u0448\u0449\u0437\u0445\u044a\u0444\u044b\u0432\u0430\u043f\u0440\u043e\u043b\u0434\u0436\u044d\u044f\u0447\u0441\u043c\u0438\u0442\u044c\u0431\u044e\u0419\u0426\u0423\u041a\u0415\u041d\u0413\u0428\u0429\u0417\u0425\u042a\u0424\u042b\u0412\u0410\u041f\u0420\u041e\u041b\u0414\u0416\u042d\u042f\u0427\u0421\u041c\u0418\u0422\u042c\u0411\u042e]')
only_chars = re.compile('[A-Za-z]')


def retrieve_model(t):
    model_descr = ""
    for token in t:
        token_ = re.sub(non_chars, "", token)
        # if re.search(only_chars, token_):
        model_descr += token_
    return model_descr.lower()

# nltk stemmer is super slow,
# I advise to run this code using pypy instead of regular python
# with pypy it took around 30 minutes to run


class Tokenizer(object):
    """
        For a given language it
         - splits the text into tokens
         - applies snowball stemmer

        Main method = @get_tokens
    """
    def __init__(self, language='russian'):
        self.stopwords = set(stopwords.words(language)).union('. , ? ! ( ) ` * ^ $ # @ + - ~ : ; < > ='.split())
        self.stemmer = SnowballStemmer('russian')

    def get_tokens_and_model(self, s):
        """
            :param s
            :type str

            :return list of str
            list of stemmed tokens
        """
        model = []
        text = []
        for word in self._str2tokens(s):
            token, is_model = self._process_token(word)
            if token is not None:
                if is_model:
                    model.append(token)
                else:
                    text.append(token)

        return " ".join(text), "".join(model)

    def _str2tokens(self, s):
        return list(set(word_tokenize(s.lower())).difference(self.stopwords))

    def _process_token(self, token):
        token_ = re.sub(non_chars, "", token)
        if token_:
            return token_, 1
        else:
            token_ = re.sub(non_russian, "", token)
            if token_:
                return self.stemmer.stem(token).encode('utf-8'), 0
            else:
                return None, None


class DataGenerator(object):
    """
    Transforms each string to list of tokens and yields @chunksize of data
    Use `for` loop to iterate
    Params:
    :param filename
    :type str
    :param chunksize
    :type int
    """
    def __init__(self, file_in, column='description', id='itemID', chunksize=1000, file_out=None, rebuild=True):
        self.file_in = file_in
        self.filename = splitext(basename(file_in))[0]
        self.column = column
        self.id = id
        self.chunksize = chunksize
        self.tokenizer = Tokenizer()

        if file_out is None:
            self.file_out = "tmp/{}_tokens.csv".format(self.filename)
        else:
            self.file_out = file_out

        if rebuild:
            rows_iterator = self._get_rows_iter()
            start = datetime.now()
            with open(self.file_out, "w") as f:
                wr = writer(f)
                for i, row in enumerate(rows_iterator):
                    wr.writerow(row)
                    if not i % 100000:
                        print('tokenized {} {}'.format(i, datetime.now() - start))

    def __iter__(self):
        rows_iterator = self._get_rows_iter(read_from_file_out=True)
        for first in rows_iterator:
            if self.chunksize > 1:
                yield chain([first], islice(rows_iterator, self.chunksize - 1))
            else:
                yield first

    def _get_rows_iter(self, read_from_file_out=False):
        if read_from_file_out:
            with open(self.file_out) as f:
                rdr = reader(f)
                for row in rdr:
                    yield unicode(row[1], 'utf-8').split()
        else:
            with open(self.file_in) as f:
                rdr = DictReader(f)
                for row in rdr:
                    yield [row[self.id]] + list(self.tokenizer.get_tokens_and_model(unicode(row[self.column], 'utf-8')))


def merge_to_pairs(prefix):
    import pandas as pd
    pairs = pd.read_csv('data/ItemPairs_{}.csv'.format(prefix))
    data1 = pd.read_csv('tmp/{}_description_tokens.csv'.format(prefix), header=None, index_col=0)
    data2 = pd.read_csv('tmp/{}_title_tokens.csv'.format(prefix), header=None, index_col=0)
    data1.columns = ["d_text", "d_model"]
    data2.columns = ["t_text", "t_model"]
    pairs.merge(
        data1, how='left', left_on='itemID_1', right_index=True
    ).merge(
        data1, how='left', left_on='itemID_2', right_index=True, suffixes=('_1', '_2')
    ).merge(
        data2, how='left', left_on='itemID_1', right_index=True
    ).merge(
        data2, how='left', left_on='itemID_2', right_index=True, suffixes=('_1', '_2')
    ).to_csv('tmp/ItemPairs_{}.csv'.format(prefix), index=False)


if __name__ == '__main__':
    tokens_stream = DataGenerator(
        file_in='data/ItemInfo_test.csv',
        column='title',
        id='itemID',
        chunksize=1,
        file_out='tmp/test_title_tokens.csv',
        rebuild=True
    )

    tokens_stream = DataGenerator(
        file_in='data/ItemInfo_test.csv',
        column='description',
        id='itemID',
        chunksize=1,
        file_out='tmp/test_description_tokens.csv',
        rebuild=True
    )

    tokens_stream = DataGenerator(
        file_in='data/ItemInfo_train.csv',
        column='title',
        id='itemID',
        chunksize=1,
        file_out='tmp/train_title_tokens.csv',
        rebuild=True
    )

    tokens_stream = DataGenerator(
        file_in='data/ItemInfo_train.csv',
        column='description',
        id='itemID',
        chunksize=1,
        file_out='tmp/train_description_tokens.csv',
        rebuild=True
    )

    # merge_to_pairs('train')
    # merge_to_pairs('test')

    # then run on bash
    #  cat tmp/train_description_tokens.csv tmp/test_description_tokens.csv > tmp/description_tokens.csv
