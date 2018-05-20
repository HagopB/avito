import os
import pickle
import pandas as pd

from keras.preprocessing import sequence 
from autoconfig import AutoConfig

config_file_name = "config.ini"
script_path = os.getcwd()


conf_file = os.path.join(script_path, config_file_name)
conf = AutoConfig(conf_file)

train_path = os.path.join(conf.path.input_path, "train.csv.zip")
test_path = os.path.join(conf.path.input_path, "test.csv.zip")
conf.path.word_vectors_path = os.path.join(conf.path.input_path, "wiki.ru.vec") # bunch of russian w2v http://rusvectores.org/en/models/

train = pd.read_csv(train_path,compression="zip", nrows = conf.data_prep.nrows)
test = pd.read_csv(test_path,compression="zip", nrows = conf.data_prep.nrows)

from pipeline import Pipeline

pi = Pipeline(conf, train, test)
vocab, mat, train_cont, test_cont, target = pi.pipe()

print("saving preprocessing in {}".format(conf.path.tmp_path))
if not os.path.exists(conf.path.tmp_path):
	os.mkdir(conf.path.tmp_path)

pickle.dump(vocab, open(os.path.join(conf.path.tmp_path, "vocab"), "wb"))
pickle.dump(mat, open(os.path.join(conf.path.tmp_path, "mat"), "wb"))
pickle.dump(train_cont, open(os.path.join(conf.path.tmp_path, "train_cont"), "wb"))
pickle.dump(test_cont, open(os.path.join(conf.path.tmp_path, "test_cont"), "wb"))
pickle.dump(target, open(os.path.join(conf.path.tmp_path, "target"), "wb"))

