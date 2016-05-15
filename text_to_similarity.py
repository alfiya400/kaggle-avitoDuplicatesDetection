from csv import DictReader, reader, writer
import time

from gensim import models, corpora
from gensim.matutils import cossim
import pandas as pd

from text_to_tokens import Tokenizer
from tokens_to_lsi import timeit


tokenizer = Tokenizer()
dictionary = corpora.Dictionary.load('tmp/train.dict')
tfidf = models.TfidfModel.load('tmp/train.tfidf')
lsi = models.LsiModel.load('tmp/model.lsi')
tfidf_corpus = corpora.MmCorpus('tmp/train_tfidf_corpus.mm')


def tokens_to_lsi(s):
    return lsi[tfidf[dictionary.doc2bow(s)]]


def description_similarity(d1, d2):
    if not d1 or not d2:
        return -2
    else:
        return cossim(
            tokens_to_lsi(d1),
            tokens_to_lsi(d2)
        )


def title_similarity(t1, t2):
    if not t1 or not t2:
        return -2
    else:
        return cossim(
            tokens_to_lsi(t1),
            tokens_to_lsi(t2)
        )


@timeit
def load_description(filepath):
    ind = []
    tokens = []
    with open(filepath) as f:
        r = reader(f)
        for row in r:
            ind.append(int(row[0]))
            tokens.append(row[1:])
    return pd.Series(tokens, index=ind)

if __name__ == '__main__':
    for _ in lsi.print_topics(num_topics=15):
        print(_)

    # load description
    description = load_description('tmp/train_description_tokens.csv')
    print(description.head(5))

    # load title
    title = pd.read_csv(
        'data/ItemInfo_train.csv', index_col='itemID',
        usecols=['itemID', 'title'], squeeze=True
    )
    print(title.head(5))

    # calc similarities
    ts = time.time()
    text_similarity = []
    with open('tmp/train_text_similarity.csv', "w") as f_out:
        w = writer(f_out)
        with open('data/ItemPairs_train.csv') as f:

            dict_reader = DictReader(f)
            for i, row in enumerate(dict_reader):
                i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                title1 = tokenizer.get_tokens(unicode(title[i1], 'utf-8'))
                title2 = tokenizer.get_tokens(unicode(title[i2], 'utf-8'))
                text_similarity.append(
                    [
                        description_similarity(description[i1], description[i2]),
                        title_similarity(title1, title2)
                    ]
                )
                if not i % 10000:
                    w.writerows(text_similarity)
                    text_similarity = []
                    print('{} {}'.format(i, time.time() - ts))

        w.writerows(text_similarity)
