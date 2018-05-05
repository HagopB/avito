import codecs
import re
import os
import numpy as np
import pandas as pd

from collections import Counter
from keras.preprocessing import sequence, text
from nltk.tokenize.toktok import ToktokTokenizer # tokenizer tested on russian
from nltk import sent_tokenize # should be multilingual


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
        spec = "/"

        regex = re.compile('[%s]' % re.escape(punct))
        regexspec = re.compile('[%s]' % re.escape(spec))
        toktok = ToktokTokenizer()
        res = data[column].fillna("__NA__").map(lambda sent: regex.sub("", sent))\
                                            .map(lambda sent: regexspec.sub(" ", sent))\
                                            .map(lambda sent: toktok.tokenize(sent.lower())).tolist()
        return res



    def create_vocab(self, sample_tok, max_word=25000, max_char=125, word_lower=True):
        UNK = "_UNK_"
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
        char_vocab = {t[0] : i for i,t in enumerate(word_counter.most_common(max_word-1))}
        char_vocab[UNK] = len(char_vocab.keys()) + 1
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

    def build_emb(self, vectors, vocab, embdim = 300):
        mat = []
        no_vectors = {}
        embedding_matrix = np.zeros((len(vocab.keys()) +1, embdim)) 
        c=0
        for i, (word, idx) in enumerate(vocab.items()):
                vect = vectors.get(word)

                if vect is not None and len(vect)>0:
                    embedding_matrix[i] = vect
                    c+=1
                else:
                    no_vectors[word] = idx
        print("{} words were found over a vocab of {} which is a ratio of {}"\
              .format(c, len(vocab.items()), round(c/len(vocab), 2) ))
        
        return embedding_matrix, no_vectors