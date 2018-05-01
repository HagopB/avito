from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os
from multiprocessing import Pool
import glob

###########################################
#--------------- FUNCTIONS ----------------
###########################################

def _fix_image_size(image, expected_pixels=2E6):
    """ fix images size
    :image param:
    :expected_pixels param optional:
    """
    ratio = float(expected_pixels) / float(image.shape[0] * image.shape[1])
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

def brightness(image):
    """ Compute image's perceived brightness
    :image param:
    """
    r, g, b = np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])
    return np.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def bluriness(image):
    """ computes the variance of the Laplacian transorm
    :image param:
    """
    image = _fix_image_size(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process(img_path):
    """ Main processing function
    :img_path param:
    """
    image = np.asarray(Image.open(img_path))
    response = [brightness(image), bluriness(image)]
    return response

def process_zipfile(img_index):
    """ Open image from a given zipfile and pre-processed it """
    with myzip.open(img_index) as myimg:
        img = np.asarray(Image.open(myimg))
        response =  {'image' : img_index.split('/')[-1].split['.'][0],
                     'brightness' : brightness(image),
                     'bluriness' : bluriness(image)}
        return response

###########################################
#------------------ MAIN ------------------
###########################################

if __name__ == '__main__':

    ZIPFILE =  './data/train_jpg.zip'
    BATCH_SIZE = 5000
    N_JOBS = 4

    scores = []
    errors = []

    with ZipFile(ZIP_PATH) as myzip:
        files_in_zip = myzip.namelist()
        steps = len(files_in_zip[1:])/BATCH_SIZE

        # getting all batch indexes
        batches = (files_in_zip[i:i+BATCH_SIZE] for i in range(1, len(files_in_zip), BATCH_SIZE)

        # multiprocessing
        p = Pool(N_JOBS)
        for idx, batch in enumerate(batches):
            print('-'*10, ' Processing batch: {}/{} '.format(idx, steps), '-'*10)
            r_tmp = list(tqdm(p.imap(process_zipfile, batch)))
            scores.extend(r_tmp)
            pickle.dump(r_tmp,open('./data/img_quality_tmp.pkl', 'wb'))

        pickle.dump(scores, open('./data/img_quality.pkl', 'wb'))
