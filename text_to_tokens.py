from csv import DictReader, writer, reader
from itertools import chain, islice
from os.path import basename, splitext
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer

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
        self.stopwords = set(stopwords.words(language)).union('. , ? ! ( )'.split())
        self.stemmer = SnowballStemmer('russian')

    def get_tokens(self, s):
        """
            :param s
            :type str

            :return list of str
            list of stemmed tokens
        """
        return map(self._process_token, self._str2tokens(s))

    def _str2tokens(self, s):
        return list(set(word_tokenize(s.lower())).difference(self.stopwords))

    def _process_token(self, token):
        return self.stemmer.stem(token).encode('utf-8')


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
                    yield row[1:]
        else:
            with open(self.file_in) as f:
                rdr = DictReader(f)
                for row in rdr:
                    yield [row[self.id]] + self.tokenizer.get_tokens(unicode(row[self.column], 'utf-8'))


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
    # then run on bash
    #  cat tmp/train_description_tokens.csv tmp/test_description_tokens.csv > tmp/description_tokens.csv
