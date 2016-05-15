import logging
import sys
import time

from gensim import corpora, models

from text_to_tokens import DataGenerator

# setup gensim logging to print to stdout
root = logging.getLogger('gensim.models.tfidfmodel')
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


def timeit(method):
    """
    Decorator to log running time for @method
    :param method: func object
    :return: @method result
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts)
        return result

    return timed


class Dictionary(object):
    """
    Init dict and save it to tmp
    """
    def __init__(self, tokens_stream=None, prefix='train', rebuild=True):
        self.tokens_stream = tokens_stream
        self.output_file = 'tmp/{}.dict'.format(prefix)
        if rebuild:
            self._init_dictionary()
        self.dictionary = corpora.Dictionary.load(self.output_file)

    @timeit
    def _init_dictionary(self):
        dictionary = corpora.Dictionary()
        for chunk in self.tokens_stream:
            dictionary.add_documents([chunk])
        dictionary.filter_extremes(no_below=50, no_above=0.8, keep_n=None)
        dictionary.save(self.output_file)

    def doc2bow(self, s):
        return self.dictionary.doc2bow(s)


class TfidfCorpus(object):
    """
    Init gensim tfidf model, save it to tmp/<prefix>.tfidf
    and serialize the corpus to tmp/<prefix>_tfidf_corpus.mm file
    """
    def __init__(self, tokens_stream, dictionary=None, prefix='train', rebuild_model=True, rebuild_corpus=True):

        """
        Initialize gensim TfIdf model and serialize the corpus
        :param tokens_stream: iter
            iterator that returns list of tokens, could be DataGenerator object
        :param dictionary: dictionary to use for gensim tfIdf model
            gensim Dictionary or any other object that supports doc2bow
        :param prefix: str
            prefix used into a filename where the tfIdf model will be saved
        :param rebuild: bool
            If True - rebuild the model and saves to output file
            If False - loads the model from output file
        """
        self.tokens_stream = tokens_stream
        self.output_file = 'tmp/{}_tfidf_corpus.mm'.format(prefix)
        self.output_model_file = 'tmp/{}.tfidf'.format(prefix)

        if dictionary is None:
            self.dictionary = Dictionary(tokens_stream, prefix=prefix, rebuild=True)
        else:
            self.dictionary = Dictionary(prefix=prefix, rebuild=False)
        print(self.dictionary.dictionary)

        if rebuild_model:
            self._init_model()

        if rebuild_corpus:
            self._init_corpus()
        self.corpus = corpora.MmCorpus(self.output_file)
        print(self.corpus)

    def __iter__(self):
        return self.corpus.__iter__()

    @timeit
    def _init_model(self):
        tfidf_model = models.TfidfModel(self._corpus_iter(), normalize=True)
        tfidf_model.save(self.output_model_file)
        del tfidf_model

    @timeit
    def _init_corpus(self):
        tfidf_model = models.TfidfModel.load(self.output_model_file)
        corpora.MmCorpus.serialize(self.output_file, tfidf_model[self._corpus_iter()])
        del tfidf_model

    def _corpus_iter(self):
        for row in self.tokens_stream:
            yield self.dictionary.doc2bow(row)

if __name__ == '__main__':
    tokens_stream = DataGenerator(
        file_in='data/ItemInfo_train.csv',
        column='description',
        id='itemID',
        chunksize=1,
        file_out='tmp/train_description_tokens.csv',
        rebuild=False
    )
    # dict_ = Dictionary(tokens_stream)
    tfidf = TfidfCorpus(tokens_stream, dictionary='tmp/train.dict')
    lsi = models.LsiModel(tfidf.corpus, id2word=tfidf.dictionary.dictionary, num_topics=400)
    lsi.print_topics()
    lsi.save('tmp/model.lsi')
