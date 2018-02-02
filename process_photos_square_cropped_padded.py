'''
This module takes each photo file listed in summitsdb and preprocesses it for input into cnn_train.py. The preprocessing includes:
1. reading the file using io.imread
2. confirm the file has enough pixels so that it is not a corrupt file
3. each image is made square by two different methods: cropping the longest side (e.g. if horizontal is longer than vertical, horizontal is cropped to vertical size), and by using zero padding on the shorter side (e.g. if horizontal is longer than vertical, the difference between the two is split in half, and an array of zeros with the horizontal width is added to the top and botton of the photo to make it square
4. each image is downsized to 100x100x3 pixels
5. the cropped and zero padded images, as np arrays, are saved as pickle files for later use in cnn_train.py

NOTE: After experimentation, the zero padded images provided much better results than the cropped images, so ultimately only the zero padded images were used.
'''

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
import psycopg2
import sys
import pickle as pickle
from skimage.transform import resize
from my_libraries import *

capstone_folder, images_folder = folders()

def doquery(cur, query, err_msg):
    '''
    INPUT:
    curs: open psql cursor
    query: string text of query to execute
    err_msg: string message if error

    OUTPUT: nothing--just executes query

    Best to close psql connection (conn) if error (error leaves conn open), cuz multiple conn open causes problems, so this function uses try to execute and commit query. If error: prints errors message, closes conn, and exits program.
    '''
    try:
        cur.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n************************* error in {}:\n{}'.format(err_msg, ex))
        cur.connection.close()
        sys.exit()


def process_image(image):
    '''
    INPUT:
    image: np.array photo image array, any size

    OUTPUT:
    image_cropped: np.array image cropped to make square, resized, normalized
    image_padded: np.array image zero-padded to make square, resized, normalized

    Process a single image for input into cnn_train.py or cnn_predict.py
    '''
    cnn_image_size = (100, 100) # Keras input image dimensions--will resize to this

    rowsize = image.shape[0]
    colsize = image.shape[1]

    def getdifs(dif):
        '''
        INPUT: dif int
        OUTPUT: dif1 int, dif2 int
        Divides dif by 2 into dif1 and dif2
        If dif is an odd number, dif2 = dif1 + 1
        '''
        dif1 = int(dif / 2)
        if dif % 2 == 0:
            dif2 = dif1
        else:
            dif2 = dif - dif1
        return dif1, dif2

    if rowsize > colsize:
        dif1, dif2 = getdifs(rowsize - colsize)
        image_cropped = image[dif1:rowsize-dif2]
        image_padded = np.hstack((np.zeros((rowsize, dif1, 3)), image, np.zeros((rowsize, dif2, 3))))

    elif colsize > rowsize:
        dif1, dif2 = getdifs(colsize - rowsize)
        image_cropped = image[:,dif1:colsize-dif2]
        image_padded = np.vstack((np.zeros((dif1, colsize, 3)), image, np.zeros((dif2, colsize, 3))))

    #reshape image to cnn_image_size for Keras/tensorflow
    #mode='constant': Pads with a constant value
    #clip=True: Clips the range of output values to the range of input values, i.e. will maintain range 0-255
    image_cropped = resize(image_cropped, cnn_image_size, clip=True, mode='constant')
    image_padded = resize(image_padded, cnn_image_size, clip=True, mode='constant')

    #normalize RGB data to range 0 to 1
    image_cropped = image_cropped.astype('float32') / 255
    image_padded = image_padded.astype('float32') / 255

    return image_cropped, image_padded

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

    images_cropped =[]
    images_padded = []
    labels_type_str = []
    labels_state = []
    errors = []

    #loop through each image in DB
    for rownum, row in enumerate(db_data):

        if rownum % 100 == 0:
            print("row# {}".format(rownum)) #to show progress

        summit_id, image_id, filename, type_str, state = row

        try:
            image = io.imread(images_folder + filename) #image is a numpy arry
        except Exception as ex:
            print("error reading summit_id {}, image_id {}".format(summit_id, image_id))
            errors.append((summit_id, image_id, "read"))

        if image.shape[0] < 20 or image.shape[1] < 20:
            print("summit_id {}, image_id {}, image.shape ={})".format(summit_id, image_id, image.shape))
            errors.append((summit_id, image_id, "size"))

        image_cropped, image_padded = process_image(image)
        images_cropped.append(image_cropped)
        images_padded.append(image_padded)

        labels_type_str.append(type_str)
        labels_state.append(state)

    #convert lists to np.arrays
    images_cropped = np.array(images_cropped)
    images_padded = np.array(images_padded)
    labels_type_str = np.array(labels_type_str)
    labels_state = np.array(labels_state)

    # save processed features and labels to pickle files
    # protocol=pickle.HIGHEST_PROTOCOL allows for files > 4GB
    with open(capstone_folder + "pickled_images_labels/images_cropped.pkl", 'wb') as f:
        pickle.dump(images_cropped, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(capstone_folder + "pickled_images_labels/images_padded.pkl", 'wb') as f:
        pickle.dump(images_padded, f, protocol=pickle.HIGHEST_PROTOCOL)
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
