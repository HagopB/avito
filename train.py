import os
import pickle
import pandas as pd
import numpy as np

from keras.preprocessing import sequence 
from autoconfig import AutoConfig

config_file_name = "config.ini"
script_path = os.getcwd()


conf_file = os.path.join(script_path, config_file_name)
conf = AutoConfig(conf_file)


print("loading data...")
vocab = pickle.load(open(os.path.join(conf.path.tmp_path, "vocab"), "rb"))
mat = pickle.load(open(os.path.join(conf.path.tmp_path, "mat"), "rb"))
train_cont = pickle.load(open(os.path.join(conf.path.tmp_path, "train_cont"), "rb"))
test_cont = pickle.load(open(os.path.join(conf.path.tmp_path, "test_cont"), "rb"))
target = pickle.load(open(os.path.join(conf.path.tmp_path, "target"), "rb"))

pad_value = len(vocab) 


padded_words = sequence.pad_sequences(train_cont['indexes']["word_id"],
      maxlen = conf.data_prep.pad_len_word, 
      value = pad_value)
padded_test = sequence.pad_sequences(test_cont['indexes']["word_id"],
      maxlen = conf.data_prep.pad_len_word,
      value = pad_value)

print("Setting category embdeddings dimensions with ratio {}".format(conf.modelling.emb_ratio))

cat_dim = round(np.unique(train_cont["cat_d"]["cat_data"][0]).shape[0] * conf.modelling.emb_ratio)
city_dim = round(np.unique(train_cont["cat_d"]["city_data"][0]).shape[0] * conf.modelling.emb_ratio)
day_dim = round(np.unique(train_cont["cat_d"]["day_data"][0]).shape[0] * conf.modelling.emb_ratio)
img_dim = round(np.unique(train_cont["cat_d"]["image_data"][0]).shape[0] * conf.modelling.emb_ratio)
parent_dim = round(np.unique(train_cont["cat_d"]["parent_data"][0]).shape[0] * conf.modelling.emb_ratio)
region_dim = round(np.unique(train_cont["cat_d"]["region_data"][0]).shape[0] * conf.modelling.emb_ratio)
user_dim = round(np.unique(train_cont["cat_d"]["user_data"][0]).shape[0] * conf.modelling.emb_ratio)


import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Reshape
from keras.layers.merge import concatenate
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D , GlobalAveragePooling1D, AveragePooling1D
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.initializers import RandomUniform

import numpy as np


def CatEmbLayer(nb_category, CATEGORY_EMB_DIM,
                input_of_layer, drop_out = 0.2):
    uniform = RandomUniform(
                        minval=-np.sqrt(3/CATEGORY_EMB_DIM),  # Suggested by
                        maxval= np.sqrt(3/CATEGORY_EMB_DIM))
    
    category_emb = Embedding(input_dim=nb_category,
                        output_dim = CATEGORY_EMB_DIM,
                        input_length = 1,
                        trainable = True, 
                        embeddings_initializer=uniform )(input_of_layer)
    category_emb = Reshape((CATEGORY_EMB_DIM, ))(category_emb)
    category_emb = Dropout(drop_out)(category_emb)
    return category_emb

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_pred-y_true, 2)))


##### MODEL
text_input = Input((conf.data_prep.pad_len_word, ))
other_feat_input = Input((train_cont["other_feat"][0].shape[1], ))

category_input = Input((1, )) #https://arxiv.org/abs/1604.06737
sub_category_input = Input((1, ))
parent_cat_input = Input((1, ))
region_cat_input = Input((1, ))
city_cat_input = Input((1, ))
image_cat_input = Input((1, ))
user_cat_input = Input((1, ))
day_cat_input = Input((1, ))


m = Embedding(input_dim = conf.data_prep.max_vocab + 2 ,
              output_dim =  conf.data_prep.emb_dim,
              weights=[mat],
              input_length=conf.data_prep.pad_len_word,
              trainable=False)(text_input)

m = Dropout(conf.modelling.first_dropout)(m)

m = Conv1D(conf.modelling.num_filters_first,
           conf.modelling.filter_size,
           activation='relu', padding='same',
           dilation_rate = conf.modelling.dilation_rate)(m)

m = AveragePooling1D(2)(m)

m = Conv1D(conf.modelling.num_filters_second, conf.modelling.filter_size,
           activation='relu',
           padding='same',
           dilation_rate = conf.modelling.dilation_rate)(m)

m = GlobalAveragePooling1D()(m)

category_emb = CatEmbLayer(train_cont['cat_s']['cat'], cat_dim, category_input)
parent_cat_emb = CatEmbLayer(train_cont['cat_s']["parent_cat"], parent_dim, parent_cat_input)
region_cat_emb = CatEmbLayer(train_cont['cat_s']["region_cat"], region_dim, region_cat_input)
city_cat_emb = CatEmbLayer(train_cont['cat_s']["city_cat"], city_dim, city_cat_input)
image_cat_emb = CatEmbLayer(train_cont['cat_s']["image_cat"], img_dim, image_cat_input)
user_cat_emb = CatEmbLayer(train_cont['cat_s']["user_cat"], user_dim, user_cat_input)
day_cat_emb = CatEmbLayer(train_cont['cat_s']["day_cat"], day_dim, day_cat_input)

m = concatenate([category_emb,
                 parent_cat_emb,
                 region_cat_emb,
                 city_cat_emb,
                 image_cat_emb,
                 user_cat_emb,
                 day_cat_emb,
                 other_feat_input,
                 m], axis = 1)

m = BatchNormalization()(m)

m = Dense(conf.modelling.first_dense, activation='relu')(m)

m = Dropout(conf.modelling.last_dropout)(m)

m = Dense(conf.modelling.second_dense, activation='relu', kernel_regularizer=regularizers.l2(conf.modelling.weight_decay))(m)

output = Dense(1)(m)

inputs = [category_input, parent_cat_input, region_cat_input, city_cat_input, image_cat_input, user_cat_input,
          day_cat_input, other_feat_input, text_input]

model = Model(inputs=inputs , 
              outputs= output,
              name='sec_model')

adam = optimizers.Adam(lr=conf.optimizer.lr, beta_1=conf.optimizer.beta_1,
                       beta_2=conf.optimizer.beta_2,
                       decay=conf.optimizer.decay, clipnorm=conf.optimizer.clip)

model.compile(loss='mean_squared_error', #mean squared error might drop to 0 faster than expected
              optimizer=adam,
              metrics=[rmse, "mae"])
#model.summary()

#early_stopping = EarlyStopping(monitor='val_rmse', min_delta=0.01, patience=7, verbose=1)
#callbacks_list = [early_stopping]

data = train_cont['cat_d']['cat_data'] + train_cont['cat_d']["parent_data"] +\
       train_cont['cat_d']["region_data"] + train_cont['cat_d']["city_data"] + train_cont['cat_d']["image_data"] +\
       train_cont['cat_d']["user_data"] + train_cont['cat_d']["day_data"] + train_cont["other_feat"]  + \
       [padded_words] 

hist = model.fit(data,
                 target,
                 batch_size=conf.modelling.batch_size,
                 epochs= conf.modelling.num_epochs, 
                 validation_split = 0.1,
                 shuffle=True,
                 verbose=2, 
                # callbacks = callbacks_list
                )


preds = model.predict(
                 test_cont['cat_d']['cat_data'] + test_cont['cat_d']["parent_data"] +\
                 test_cont['cat_d']["region_data"] + test_cont['cat_d']["city_data"] + test_cont['cat_d']["image_data"] +\
                 test_cont['cat_d']["user_data"] + test_cont['cat_d']["day_data"] + test_cont["other_feat"]  + \
                 [padded_test] ,
)




from utils import write_output, plot_history

write_output(preds, conf)
plot_history(hist, conf, preds, target, save = True)