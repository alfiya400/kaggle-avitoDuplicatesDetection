from csv import DictReader, reader, writer
import time

from gensim import models, corpora
from gensim.matutils import cossim
import pandas as pd

from tokens_to_lsi import timeit


# dictionary = corpora.Dictionary.load('tmp/train.dict')
# tfidf = models.TfidfModel.load('tmp/train.tfidf')
# lsi = models.LsiModel.load('tmp/model.lsi')
model = models.Word2Vec.load('tmp/model.w2v')
# def tokens_to_lsi(s):
#     return lsi[tfidf[dictionary.doc2bow(s)]]


def description_similarity(d1, d2):
    if not d1 or not d2:
        return -3
    elif not d1 or not d2:
        return -2
    else:
        d1 = [w for w in unicode(d1, 'utf-8').split() if w in model.vocab]
        d2 = [w for w in unicode(d2, 'utf-8').split() if w in model.vocab]
        return model.n_similarity(d1, d2) if d1 and d2 else 0


def title_similarity(t1, t2):
    if not t1 and not t2:
        return -3
    elif not t1 or not t2:
        return -2
    else:
        t1 = [w for w in unicode(t1, 'utf-8').split() if w in model.vocab]
        t2 = [w for w in unicode(t2, 'utf-8').split() if w in model.vocab]
        return model.n_similarity(t1, t2) if t1 and t2 else 0


@timeit
def load_tokens(filepath):
    ind = []
    tokens = []
    with open(filepath) as f:
        r = reader(f)
        for row in r:
            ind.append(int(row[0]))
            tokens.append(row[1:])
    return pd.Series(tokens, index=ind)

if __name__ == '__main__':
    # for _ in lsi.print_topics(num_topics=15):
    #     print(_)

    for prefix in ['train', 'test']:

        # calc similarities
        ts = time.time()
        text_similarity = []
        with open('tmp/{}_text_similarity.csv'.format(prefix), "w") as f_out:
            w = writer(f_out)
            with open('tmp/ItemPairs_{}.csv'.format(prefix)) as f:

                dict_reader = DictReader(f)
                for i, row in enumerate(dict_reader):
                    i1, i2 = int(row['itemID_1']), int(row['itemID_2'])
                    text_similarity.append(
                        [
                            description_similarity(row['d_text_1'], row['d_text_2']),
                            title_similarity(row['t_text_1'], row['t_text_2'])
                        ]
                    )
                    if not i % 100000:
                        w.writerows(text_similarity)
                        text_similarity = []
                        print('{} {}'.format(i, time.time() - ts))

            w.writerows(text_similarity)
