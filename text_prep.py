import codecs
import re
import os
import numpy as np
import pandas as pd

from collections import Counter
from keras.preprocessing import sequence, text
from nltk.tokenize.toktok import ToktokTokenizer # tokenizer tested on russian
from nltk import sent_tokenize # should be multilingual
from string import punctuation
from nltk import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText


russian_stops = stopwords.open('russian').read().splitlines()
def filter_params(param):
    return [w.lower() for w in param.split() if w.lower() not in russian_stops ]


class Preparator():
    def __init__(self):
        pass

    def load_vectors(self, word_vectors_path, number= 40000):
        ## load vectors ranked by frequency
        with codecs.open(word_vectors_path, encoding='utf-8') as f:
            vectors = {}
            for i, line in enumerate(f):
                line = line.split()
                try:
                    vectors[line[0]] = np.asarray(line[1:], dtype='float32')
                except:
                    continue
                if i > number:
                    break
        return vectors




    def clean_col(self, column, data):
        punct = "!#$«%&\'.()*+-<=>?@[\\]^_`°{|}"
        spec = "/\n"
        digits = "\d"
        regex = re.compile('[%s]' % re.escape(punct))
        regexspec = re.compile('[%s]' % re.escape(spec))
        digits = re.compile(digits)
        toktok = ToktokTokenizer()
        res = data[column].fillna("__NA__").map(lambda sent: regex.sub("", sent))\
                                            .map(lambda sent: digits.sub("#", sent))\
                                            .map(lambda sent: regexspec.sub(" ", sent))\
                                            .map(lambda sent: toktok.tokenize(sent.lower())).tolist()
        return res



    def create_vocab(self, sample_tok, max_word=25000, max_char=125, word_lower=True):
        UNK = "_UNK_"
        PAD = "_PAD_"
        # vocab
        word_counter = Counter()
        char_counter = Counter()

        # traverse
        for sentence in sample_tok:
            for tok in sentence:
                for c in tok:
                    char_counter[c] += 1
                if word_lower:
                    tok = tok.lower()
                    word_counter[tok] += 1

        # vocab size limit
        word_vocab = {t[0] : i for i,t in enumerate(word_counter.most_common(max_word-1))}
        word_vocab[UNK] = len(word_vocab.keys()) + 1
        word_vocab[PAD] = len(word_vocab.keys()) + 2

        char_vocab = {t[0] : i for i,t in enumerate(word_counter.most_common(max_word-1))}
        char_vocab[UNK] = len(char_vocab.keys()) + 1
        char_vocab[PAD] = len(char_vocab.keys()) + 2
        return word_vocab, char_vocab

    from itertools import chain

    def flatten(l):
        return list(chain.from_iterable(l))

    def vocab_lookup(self, sentences , word_idx, char_idx, word_lower=True):
            word_vec = []
            char_vec = []
            UNK = "_UNK_"
            # traverse
            for tok_list in sentences:
                idx_tok_list = []
                char_list = []
                for tok in tok_list:
                    # char level
                    idx_char_list = []
                    for c in tok:
                        idx_char_list.append(char_idx.get(c, char_idx[UNK]))
                    char_list.append(idx_char_list)

                    # word lower
                    if word_lower:
                        tok = tok.lower()
                    idx_tok_list.append(word_idx.get(tok, word_idx[UNK]))

                word_vec.append(idx_tok_list)
                char_vec.append(char_list)

            # sample
            sample = dict()
            sample['word_id'] = word_vec
            sample['char_id'] = char_vec
            return sample

    def build_emb(self, vectors, vocab, embdim = 300, model_path = None):
        if model_path:
            ft = FastText.load_fasttext_format(model_path)

        mat = []
        no_vectors = {}
        embedding_matrix = np.zeros((len(vocab.keys()) + 1, embdim)) 
        c=0
        for i, (word, idx) in enumerate(vocab.items()):
                if model_path:
                    try:
                        vect = ft[word]
                    except:
                        vect = None
                else:
                    vect = vectors.get(word)

                if vect is not None and len(vect)>0:
                    embedding_matrix[i] = vect
                    c+=1
                else:
                    no_vectors[word] = idx
        print("{} words were found over a vocab of {} which is a ratio of {}"\
              .format(c, len(vocab.items()), round(c/len(vocab), 2) ))
        
        return embedding_matrix, no_vectors

    def filter_params(param):
            return [w.lower() for w in param.split() if w.lower() not in russian_stops]

    def prep_data(self, dataset):
        na_cols = ["description", "title", "param_1", "param_2", "param_3"]
        dataset[na_cols] = dataset[na_cols].fillna('__NA__')
        dataset[['price', "image_top_1"]] = dataset[['price', "image_top_1"]].fillna(dataset['price'].median())
        regex = re.compile('[%s]' % re.escape(punctuation))
        struct = "/\n✔;"
        struct = re.compile('[%s]' % re.escape(struct))


        dataset['price'] = dataset['price'].fillna(dataset['price'].median())
        dataset[["description", "title"]] = dataset[["description", "title"]].fillna('__NA__')
        dataset["text"] = dataset.apply(lambda x: x["title"] + " END_DESC " + x["description"], axis=1)
        dataset["text_len"] = dataset.text.map(len)
        dataset["nb_words"] = dataset.text.map(lambda x: len(x.split()))
        dataset["nb_sents"] = dataset.text.map(lambda x: len(sent_tokenize(x)))
        dataset["nb_punct"] = dataset.text.map(lambda x: len(regex.findall(x)))
        dataset["words_price"] = dataset.apply(lambda x: x['price']/x["nb_words"], axis = 1 ) # longer description for more expensive products
        # ex a long description for flat is more important than for shoes..

        dataset['structure'] = dataset.text.map(lambda x: len(struct.findall(x))/len(x)) # structured text (bullet points carriage, returns etc..)
        #ex: dataset.loc[451,"description"]
        dataset["digits_count"] = dataset.text.map(lambda x: len(re.findall("\d+", x))/len(x))
        dataset["dayofweek"]= pd.DatetimeIndex(dataset.activation_date).dayofweek
        return dataset

    def encode_embedding(self, column, train, test):
        unique_cat = list(train[column].unique()) + list(test[column].unique()) 
        index2cat = {c: i for c, i in enumerate(unique_cat)}
        cat2index = {v: k for k, v in index2cat.items()}
        data_col = np.array(train[column].map(cat2index), dtype = np.int64)
        return len(unique_cat), data_col