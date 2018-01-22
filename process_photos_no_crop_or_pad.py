import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
# from PIL import Image
import psycopg2
import sys
import itertools
import pickle as pickle
from skimage.transform import resize
from my_libraries import *

capstone_folder, images_folder = folders()

def doquery(curs, query, err_msg):
    '''
    INPUT:
    curs: open psql cursor
    query: string text of query to execute
    err_msg: string message if error

    OUTPUT: nothing--just executes query

    Best to close psql connection (conn) if error (error leaves conn open), cuz multiple conn open causes problems, so this function uses try to execute and commit query. If error: prints errors message, closes conn, and exits program.
    '''
    try:
        curs.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n************************* error in {}:\n{}'.format(err_msg, ex))
        cur.connection.close()
        sys.exit()


def process_image(image, cnn_image_size):
    '''
    INPUT:
    image: np.array photo image array, any size

    OUTPUT:
    image_cropped: np.array image cropped to make square, resized, normalized
    image_padded: np.array image zero-padded to make square, resized, normalized

    Process a single image for input into cnn_train.py or cnn_predict.py
    '''
    # standard_photo_rows = 300
    # standard_photo_cols = 400

    # image = np.array([np.arange(26250)]).reshape(150,175) #TESTING

    #reshape image to cnn_image_size for Keras/tensorflow
    #mode='constant': Pads with a constant value
    #clip=True: Clips the range of output values to the range of input values, i.e. will maintain range 0-255
    image  = resize(image, cnn_image_size, clip=True, mode='constant')

    #normalize RGB data to range 0 to 1
    image = image.astype('float32') / 255

    return image

if __name__ == "__main__":
    #open cursor to summitsdb
    home = os.getenv("HOME")
    if home == '/Users/edwardwhite': #MacBook
        conn = psycopg2.connect(dbname='summitsdb', host='localhost')

    elif home == '/home/ed': #Linux
        with open(home +  '/.google/psql', 'r') as f:
            p = f.readline().strip()
        conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='ed', password=p)
        p = None
    cur = conn.cursor()

    #get data on images to process from summitsdb
    query = '''
    SELECT i.summit_id, i.image_id, i.filename, s.type_str, s.state
    FROM images i INNER JOIN summits s ON i.summit_id=s.summit_id
    ORDER BY i.image_id;
    '''
    doquery(cur, query, "select filenames")
    db_data = cur.fetchall()

    cnn_image_size = (175, 175) # Keras input image dimensions--will resize to this

    images = []
    labels_type_str = []
    labels_state = []
    errors = []

    # rownum = 0
    # row = (16669, 1, 'CA_0_16669_1.jpg', 0, 'CA')
    for rownum, row in enumerate(db_data):

        if rownum % 1000 == 0:
            print("image# {}".format(rownum)) #to show progress

        summit_id, image_id, filename, type_str, state = row

        try:
            image = io.imread(images_folder + filename) #image is a numpy arry
        except Exception as ex:
            print("error reading summit_id {}, image_id {}".format(summit_id, image_id))
            errors.append((summit_id, image_id, "read"))

        if image.shape[0] < 20 or image.shape[1] < 20:
            print("summit_id {}, image_id {}, image.shape ={})".format(summit_id, image_id, image.shape))
            errors.append((summit_id, image_id, "size"))

        image = process_image(image, cnn_image_size) # Keras input image dimensions--will resize to this

        images.append(image)

        labels_type_str.append(type_str)
        labels_state.append(state)
    print("Finished processing {} images.".format(rownum))

    #convert lists to np.arrays
    images = np.array(images)
    labels_type_str = np.array(labels_type_str)
    labels_state = np.array(labels_state)

    # save processed features and labels to pickle files
    # protocol=pickle.HIGHEST_PROTOCOL allows for files > 4GB
    with open(capstone_folder + "pickled_images_labels/images.pkl", 'wb') as f:
        pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(capstone_folder + "pickled_images_labels/labels_type_str.pkl", 'wb') as f:
        pickle.dump(labels_type_str, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(capstone_folder + "pickled_images_labels/labels_state.pkl", 'wb') as f:
        pickle.dump(labels_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(capstone_folder + "pickled_images_labels/cnn_image_size.pkl", 'wb') as f:
        pickle.dump(cnn_image_size, f, protocol=pickle.HIGHEST_PROTOCOL)


    if len(errors) > 0:
        with open(capstone_folder + "still_errors-new.pkl", 'wb') as f:
            pickle.dump(errors, f)
    else:
        print("all images downloaded OK")
