from text_prep import Preparator
import pandas as pd
import numpy as np

class Pipeline():
	def __init__(self, conf, train, test):
		self.conf = conf
		self.train = train
		self.test = test
		self.pr = Preparator()

	def pipe(self):
		print("loading vectors..")
		vectors = self.pr.load_vectors(self.conf.path.word_vectors_path,
								  self.conf.data_prep.n_vectors)

		print("preparing data...")
		self.train = self.pr.prep_data(self.train)
		self.test = self.pr.prep_data(self.test)
		

		print("cleaning text...")
		train_text = self.pr.clean_col("text", self.train)
		test_text = self.pr.clean_col("text", self.test)

		vocab, char_vocab = self.pr.create_vocab(train_text + test_text,
			self.conf.data_prep.max_vocab,			
			self.conf.data_prep.max_char)

		print("Vocab lookup...")
		train_indexed = self.pr.vocab_lookup(train_text, vocab,  char_vocab)
		test_indexed = self.pr.vocab_lookup(test_text, vocab,  char_vocab)

		mat, novect = self.pr.build_emb(vectors, vocab, self.conf.data_prep.emb_dim)

		print("Encode embedding...")
		cat_s_tr, cat_d_tr = self.prepare_cat_embeddings(self.train, self.test)
		cat_s_ts, cat_d_ts = self.prepare_cat_embeddings(self.test, self.train)
		
		other_features = ["text_len", "nb_words", "nb_sents", "nb_punct",
                  		  "words_price", "structure", "digits_count", "price"]

		other_feat_tr = np.array(self.train[other_features])
		other_feat_ts = np.array(self.test[other_features])
		
		train_cont = {}
		test_cont = {}

		train_cont["indexes"] = train_indexed
		train_cont["other_feat"] = [other_feat_tr]
		train_cont["cat_s"] = cat_s_tr
		train_cont["cat_d"] = cat_d_tr

		test_cont["indexes"] = test_indexed
		test_cont["other_feat"] = [other_feat_ts]
		test_cont['cat_s'] = cat_s_ts
		test_cont['cat_d'] = cat_d_ts

		del train_indexed, other_feat_tr, cat_s_tr, cat_d_tr, test_indexed, other_feat_ts, cat_s_ts, cat_d_ts
		target = np.array(self.train.deal_probability)

		return vocab, mat, train_cont, test_cont, target

	def prepare_cat_embeddings(self, data, test):
		to_emb = ["category_name", "parent_category_name", "region", "city",
		"image_top_1", "user_type", "dayofweek"]
		emb_cols = [self.pr.encode_embedding(col, data, test) for col in to_emb]
		sizes, datasets = zip(*emb_cols)

		datasets =[[s] for s in datasets] 
		cat_s = {}
		cat_d = {}

		cat_s["cat"], cat_s["parent_cat"], cat_s["region_cat"], cat_s["city_cat"], cat_s["image_cat"], cat_s["user_cat"], cat_s["day_cat"] = sizes
		cat_d["cat_data"], cat_d["parent_data"], cat_d["region_data"], cat_d["city_data"], cat_d["image_data"], cat_d["user_data"], cat_d["day_data"] = datasets
		return cat_s, cat_d

