import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from zipfile import ZipFile

# keras
import keras
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model

###########################################
#--------------- FUNCTIONS ----------------
###########################################

# IQA model
def IQA(weights_path):
    """ getting the NIMA IQA pre-trained model """
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)

    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(weights_path)
    return model

# iqa utils
def mean_score(scores):
    """ computes the IQA mean score """
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean

def std_score(scores):
    """ computes the IQA std score """
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

# predict 
def predict_IQA(img, model):
    """ Predict the IQA """
    img = 2*(img/255.0)-1.0 # process images
    return model.predict(img) # predict


###########################################
#------------------ MAIN ------------------
###########################################

if __name__ == '__main__':
    iqa = IQA('./inception_resnet_weights.h5')
    
    zip_path = './data/train_jpg.zip'
    scores = []
    errors = []
    
    with ZipFile(zip_path) as myzip:
        files_in_zip = myzip.namelist()
        for idx, file in enumerate(tqdm(files_in_zip[150001:])):
            try:
                with myzip.open(file) as myfile:
                    img = Image.open(myfile)
                    img = np.expand_dims(np.asarray(img.resize((224, 224), Image.ANTIALIAS)), 0)

                    # predict
                    score = predict_IQA(img, iqa)

                    # store results
                    scores.append({'image' : file.split('/')[-1],
                                   'score' : score,
                                   'mean' : mean_score(score),
                                   'std' : std_score(score)})

                    if idx % 5000 == 0:
                        pickle.dump(scores, open('./iqa_res/scores_train_tmp.pkl','wb'))
            except:
                errors.append(file)
                pass
                
pickle.dump(scores, open('./iqa_res/scores_train.pkl','wb'))
pickle.dump(errors, open('./iqa_res/errors_train.pkl','wb'))