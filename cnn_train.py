'''
This is the main module for running the convolutional neural network (CNN). It reads in the pickled data from process_photos_square_padded.py (using the padded photos, which gave the best results--images are made square using zero-padding, then resized to 100x100 pixels), filters them per the criteria in the runs_params list in the __main__ section, balances the number of rows in each class, splits the data into 80%/20% training/test sets, fits the CNN on the training data, and provides the accuracy results from the test set. The fitted models are saved in JSON files for future use in predicting individual photos.
'''
import numpy as np
np.random.seed(1000) #to get consistent results every time--used to test hyperparameter

import pandas as pd
from sklearn.utils import resample
import sys
import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow import set_random_seed #needed in addition to np.random.seed() to get consisitent results for testing hyperparameters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
import os
import datetime
from alchemy_conn import alchemy_engine
from my_libraries import *

set_random_seed(1000)

capstone_folder, images_folder = folders()
# np.random.RandomState = 1000 #does NOT give consistent results every time--use np.random.seed instead


def get_db_conn_cursor():
    '''return psql connection and cursor to the PSQL summitsdb'''
            #set up connection and cursor to psql summitsdb
    home = os.getenv("HOME")
    if home == '/Users/edwardwhite': #MacBook
        conn = psycopg2.connect(dbname='summitsdb', host='localhost')

    elif home == '/home/ed': #Linux
        with open(home +  '/.google/psql', 'r') as f:
            p = f.readline().strip()
        conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='ed', password=p)
        p = None
    return conn, conn.cursor()


def doquery(cur, query, errmsg):
    '''
    INPUT
    cur: cursor to the PSQL database
    query: SQL query string
    errmsg: string to display if an error occurs during query execution

    When an error occurs during query execution, it usually leaves the DB connection open, and having too many open connections can cause a problem. This function uses try-except and closes the connection if an error occurs.
    '''
    try:
        cur.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n********************************************* error in {}\n{}'.format(errmsg, ex))
        cur.connection.close()
        sys.exit()


class cnn_class(object):
    '''
    Generates and saves in json files fitted cnn models on images (np arrays) dataset previously processed.

    The array of images read into this class has shape (rows of images, 100, 100, 3). 100x100x3 is an image of number pixels horizontal, number pixels vertical, and 3 colors RGB--this is the tensorflow input_shape, as opposed to the theano shape of 3xhorizonalxvertial, which is not used here.
    '''
    def __init__(self, model_num):
        '''This class is run with a specific set of cnn hyperparameters, and each unique hyperparameter set is designated with a sequential integer model_num. The fitted models are saved to a filename that includes the model_num.
        '''
        self.model_num = model_num


    def filter_images(self, images, labels, *labels_filter):
        '''
        INPUT
        images: np array of images
        labels: np array of strings--e.g. either state labels (2 letter abbrev), or "mount"/"mountain"/"peak" for summit type
        *labels_filter: strings --> unique subset of labels to filter images and labels by

        OUTPUT
        np array: subset of images corresponding to labels filtered by *labels_filter
        np array: subset of labels filtered by *labels_filter, but convertered to integers starting with 0 for first item in labels_filter
        string: concatenation of each of *labels_filter separated by "_"--used in identifying which comparison was done, e.g. in the saved model's filename
        integer: len(*labels_filter), i.e. the number of classes compared
        list--strings naming each class, sorted alphabetically

        Filter images and labels per *labels_filter, and convert string labels to one hot encoding, e.g.:
        images = np.array([11, 12, 13, 14, 15])
        labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
        filter_images(images, labels, 'CO', 'NM') returns:
        np.array([11, 13, 14, 15])
        np.array([0, 0, 1, 0])
        'CO_NM'
        2
        ['CO', 'NM']
        '''
        num_classes = len(labels_filter)

        if num_classes != len(set(labels_filter)):
            print("*labels_filter must contiain a unique list, and it is not unique. TERMINATING PROGRAM.")
            sys.exit()
        classes = sorted(list(set(labels_filter))) #list of unique classes to be returned

        X = [] #images to be returned
        y = [] #converted labels to be returned
        comparison = '' #string to be returned
        classes = [] #list to be returned
        y_col_index = dict() #key=choice, value=y column index, used below

        print("Classes being compared:")
        for col_index, choice in enumerate(labels_filter):
            print(col_index, choice)
            if choice not in labels:
                print("\n\nchoice: {} is not in labels.\nMAKE SURE YOU'VE SELECTED THE CORRECT labels_filter FOR THESE labels.\nTERMINATING PROGRAM.".format(choice))
                sys.exit()

            y_col_index[choice] = col_index
            comparison += choice + "_"
            classes.append(choice)
        comparison = comparison[:-1] #drop last "_" from comparison

        for image, label in zip(images, labels):
            if label in labels_filter:
                X.append(image)
                y.append(y_col_index[label])

        return np.array(X), np.array(y), comparison, num_classes, classes


    # #TEST above
    # images = np.array([11, 12, 13, 14, 15])
    # labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
    # X, y, comparison, num_classes, classes = filter_images(None, images, labels, 'CO', 'NM')
    # print ("{}\n{}\n{}\n{}\n{}".format(X, y, comparison, num_classes, classes))


    def balance_classes(self, X, y, numrows_in_each_class, num_classes, images_or_data):
        '''
        INPUT
        X: np array--images or numerical features (elevation, isolation, prominence)
        y: np array--integer labels
        num_classes: int--# unique values in y
        numrows_in_each_class: int--desired size of each class
        images_or_data: string--if "images", upsampling includes flipping images horizontally--this flipping doesn't apply when balance
        OUTPUT:
        X, y: np.arrays with numrows_in_each_class number of rows for each class

        Makes X, y have same number of rows, numrows_in_each_class, for each class in y. if images_or_data=="images", to the extent upsampling is required, some of the new images added are flipped horizontally.
        '''
        classes = np.unique(y) #number of unique classes
        # print("classes: {}".format(classes))
        num_per_class = np.zeros((num_classes), dtype=int)
        X_class = [] #each element of list = features for that class
        y_class = [] #each element of list = labels for that class

        for i, value in enumerate(classes):
            indices = np.where(y == value)[0]
            X_class.append(X[indices]) # features for this class
            y_class.append(np.full(numrows_in_each_class, value, dtype=int)) # labels for this class i. Goal is to make each X_class[i] and y_class[i] have numrows_in_each_class each, with some of the new X_class[i] images that may be added flipped horizontally. Each y_class[i] has same labels, so order of rows within each X_class[i] doesn't matter. Each X_class[i] and y_class[i] are reassembled at end of this function in order of i's to maintain feature-label consistency.
            num_per_class[i] = indices.size

        #for each class in X, make that class have numrows_in_each_class number of rows
        for i, num in enumerate(num_per_class):
            if num == numrows_in_each_class:
                continue #already correct size, so no adjustment needed

            elif num > numrows_in_each_class:
            # downsample to numrows_in_each_class rows
                X_class[i] = resample(X_class[i], replace = False, n_samples=numrows_in_each_class) #random_state set in np.random.seed at top

            else: # num < numrows_in_each_class
                #upsize with replacement
                #since cannot resize with replacement if n_samples > existing samples, repeat resampling several times to get to numrows_in_each_class
                num_iters_upsample = int(numrows_in_each_class / num) - 1
                num_addition_upsamples = numrows_in_each_class % num

                for j in range(num_iters_upsample):
                    X_newsample = resample(X_class[i], replace = False, n_samples=num) #random_state set in np.random.seed at top

                    if j == 0:
                        X_newsamples = np.copy(X_newsample)
                    else:
                        X_newsamples = np.vstack((X_newsamples, X_newsample))

                #remainder of number rows needed to upsample
                if num_addition_upsamples > 0:
                    X_newsample = resample(X_class[i], replace = False, n_samples=num_addition_upsamples) #random_state set in np.random.seed at top

                    if num_iters_upsample == 0:
                        X_newsamples = np.copy(X_newsample)
                    else:
                        X_newsamples = np.vstack((X_newsamples, X_newsample))

                if images_or_data == "images":
                    '''If existing number of rows is less than 50% of existing rows, then the new images added will be half resampled original images, and half horizontally flipped images'''
                    if num_per_class[i] < .5 * numrows_in_each_class:
                        #split X_newsamples in 2, and apply make_different_image to half of them
                        half_num_newsamples = int(len(X_newsamples) / 2)
                        X_newsamples_half1 = X_newsamples[:half_num_newsamples]
                        X_newsamples_half2 = X_newsamples[half_num_newsamples:]
                        X_newsamples_half1.shape, X_newsamples_half2.shape

                        X_newsamples_half1 = self.make_different_images(X_newsamples_half1)
                        np.vstack((X_newsamples_half1, X_newsamples_half2))

                        X_newsamples = np.vstack((X_newsamples_half1, X_newsamples_half2))

                    else:
                        '''If existing number of rows is more than 50% of existing rows, then all new images added will be flipped horizontally'''
                        X_newsamples = self.make_different_images(X_newsamples)

                    #append new samples to existing ones
                X_class[i] = np.vstack((X_class[i], X_newsamples))


        #X_class[i] and y_class[i] now should have numrows_in_each_class rows each
        #recombine the separate classes onto one np array
        X = np.vstack((X_class[i] for i in range(num_classes)))
        y = np.vstack((np.array(y_class[i]) for i in range(num_classes))).reshape(numrows_in_each_class *  num_classes)
        print()
        print("balance_classes--after: X.shape {}, y.shape {}".format(X.shape, y.shape))
        return X, y


    def make_different_images(self, images):
        '''
        INPUT
        images: np array of images

        OUTPUT: np array of images, same shape as input, with each image flipped horizontally

        Flip each image in images horizontally and return them'''
        f = lambda x: x[::-1]
        f1 = lambda x: np.apply_along_axis(f, 1, x)
        return np.array([f1(x) for x in images])


    def one_hot_encode_labels(self, y_not_categ, num_classes):
        '''
        INPUT
        y_not_categ: np array of integer labels
        num_clases: integer representing number of unique labels

        OUTPUT: np array of one hot encoded labels

        Turn integer labels into one hot encoded labels, e.g.
        one_hot_encode_labels(np.array([0, 0, 1, 0]), 2) returns:
        np.array([1, 0], [1, 0], [0, 1], [1, 0])
        '''
        y = np.zeros((len(y_not_categ), num_classes))
        for y_row, y_val in enumerate(y_not_categ):
            y[y_row, y_val] = 1
        return np.array(y)
    # #TEST above
    # print(one_hot_encode_labels(None, np.array([0, 0, 1, 0]), 2))


    def fit_cnn(self, X_train, y_train, X_test, y_test, **params):
        '''
        INPUT
        X_train, y_train, X_test, y_test: np.arrays of X images, y labels (one hot vectors) split into train, test sets

        **params:
        num_epochs--12
        batch_size--128, 140 mine

        number of convolutional filters to use:
        num_filters--32 #32 orig, 30 mine

        size of pooling area for max pooling:
        pool_size--(3,3) #(2, 2)

        # convolution kernel size:
        kernel_size--(4,4) #(3, 3)

        #(# pixels horizontal, #pixelsvertical)
        input_shape--(100,100)

        dense--128
        dropout1--0.25
        dropout2--0.25


        OUTPUT
        no output, although fitted modesl are saved to JSON files, and accuracy results are store in model_fit_results table of summitsdb
        '''
        #create Sequential cnn in Keras
        model = Sequential()
        # 2 convolutional layers followed by a pooling layer followed by dropout
        model.add(Conv2D(filters=params['num_filters'], kernel_size=(params['kernel_size']), padding='valid', input_shape=params['input_shape']))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=params['num_filters'], kernel_size=params['kernel_size']))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=params['pool_size']))
        model.add(Dropout(params['dropout1']))

        # transition to an mlp
        model.add(Flatten())
        model.add(Dense(params['dense']))
        model.add(Activation('relu'))
        model.add(Dropout(params['dropout2']))
        model.add(Dense(params['num_classes']))
        model.add(Activation('softmax')) #provides final probability of each class

        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam', #nadam worked best in my experiments
                      metrics=['accuracy'])

        # cnn_test_accuracy = (.5, .5)
        model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['num_epochs'], verbose=1, validation_data=(X_test, y_test))

        print("Finished model.fit(). Now model.evaluate().")
        cnn_test_accuracy = 100 * model.evaluate(X_test, y_test, verbose=0)[1]
        print('Test accuracy={} <-------------------------\n'.format(cnn_test_accuracy))
        print("Finished model evaluate(). Now saving data and model.")

        #save acccuray results and model parameters in summitsdb
        conn, cur = get_db_conn_cursor()

        fn_prefix = capstone_folder + "models/"
        model_filename = "cnn_model_" + str(self.model_num) + "_" + params['comparison'] + "_by_" + params['type']

        query = "INSERT INTO model_fit_results (model_num, time_completed, cnn_test_accuracy, "

        for param in params:
            query += param + ", "

        query += 'model_filename'") VALUES ({}, TIMESTAMP WITH TIME ZONE '{}', {}, ".format(self.model_num, datetime.datetime.now(), cnn_test_accuracy)

        for value in params.values():
            if type(value) == str:
                query += "'{}', ".format(value)
            elif type(value) == tuple:
                query += "'{"
                for v in value:
                    query += "{}, ".format(v)
                query = query[:-2] #drop last ", " at end
                query += "}', "
            else:
                query += "{}, ".format(value)
        query += "'{}');".format(model_filename)

        doquery(cur, query, "INSERT INTO model_fit_results")

        cur.close()
        conn.close()

        #save model for future use
        model_filename = fn_prefix + model_filename
        try:
            print("Saving cnn model to json & h5...")
            with open(model_filename + ".json", "w") as f:
                f.write(model.to_json())
            model.save_weights(model_filename + ".h5")
            print("Done saving model to json & h5.")
        except Exception as ex:
            print("!!!!!! Error trying to save cnn model to json & h5 !!!!!!!:\n{}".format(ex))
            sys.exit()

        print("Finished saving data and model.")
        return model, cnn_test_accuracy # model, cnn_test_accuracy


    def run_GradientBoosting(self, df, by_state_or_type, numrows_in_each_class, comparison, *labels_filter):
        '''
        INPUT
        df: DataFrame containing at least state, type_str, elevation, isolation, prominence. state is two letter abbreviation, type_str is string
        by_state_or_type: string equals either "state" or "type"
        *labels_filter: list of states or types to filter data by

        OUTPUT
        Nothing is returned.

        The sklearn GradientBoostingClassifier is run on the data in the df dataframe. Similar to cnn_fit(), accuracy results are saved in the model_fit_results table in summitsdb, and the model is saved in a pickle file, and also the minimum/maximum of the data ranges are stores for future use in normalizing data for individual photos that might be examined in cnn_predict.
        '''
        #filter data per labels_filter

        X = []
        y = []
        y_col_index = dict() #key=choice, value=y column index, used below

        if by_state_or_type == "type":
            by_state_or_type = "type_str"
        for col_index, choice in enumerate(labels_filter):
            if choice not in df[by_state_or_type].values:
                print("\n\nchoice: {} is not in labels.\nMAKE SURE YOU'VE SELECTED THE CORRECT labels_filter FOR THESE labels.\nTERMINATING PROGRAM.".format(choice))
                sys.exit()

            y_col_index[choice] = col_index

        for i, df_row in df.iterrows():
            if df_row[by_state_or_type] in labels_filter:
                X.append(df_row[['elevation', 'isolation', 'prominence']])
                y.append(y_col_index[df_row[by_state_or_type]])

        X = np.array(X)
        y = np.array(y)

        num_classes = len(labels_filter)
        X, y = self.balance_classes(X, y, numrows_in_each_class, num_classes, images_or_data="data")

        #normalize features data to range 0 - 1
        X_min = X.min(axis=0) #axis=0 -> up and down columns
        X_max = X.max(axis=0)
        X = (X - X_min) / (X_max - X_min)
        minmax = [X_min, X_max]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        model = GradientBoostingClassifier(criterion='friedman_mse', loss='deviance', max_depth=7, max_features='sqrt', n_estimators=200)

        model.fit(X_train, y_train)

        print("Finished GBC model.fit(). Now model.score().")
        gbc_test_accuracy = 100 * model.score(X_test, y_test)
        print('Test accuracy={} <-------------------------\n'.format(gbc_test_accuracy))
        print("Finished model evaluate(). Now saving data and model.")

        #save results to DB
        conn, cur = get_db_conn_cursor()

        if by_state_or_type == "type_str":
            by_state_or_type = "type"
        query = '''
        UPDATE model_fit_results
        SET gbc_test_accuracy={}
        WHERE model_num={} AND comparison='{}' AND by_type='{}';
        '''.format(gbc_test_accuracy, self.model_num, comparison, by_state_or_type)
        doquery(cur, query, "UPDATE model_fit_results gbc_test_accuracy")
        # print(query)

        cur.close()
        conn.close()

        #save gbc model for future use
        model_filename = capstone_folder + "models/gbc_model_" + str(self.model_num) + "_" + comparison + "_by_" + by_state_or_type
        try:
            print("Saving gbc model to pickle...")
            with open(model_filename + ".pkl", 'wb') as f:
                pickle.dump(model, f)
            print("Done saving gbc model to pickle.")
        except Exception as ex:
            print("!!!!!! Error trying to save gbc model to pickle !!!!!!!:\n{}".format(ex))
            sys.exit()

        #save min-max for future use in normalizing data
        model_filename = capstone_folder + "models/minmax_" + str(self.model_num) + "_" + comparison + "_by_" + by_state_or_type
        try:
            print("Saving minmax to pickle...")
            with open(model_filename + ".pkl", 'wb') as f:
                pickle.dump(minmax, f)
            print("Done saving minmax to pickle.")
        except Exception as ex:
            print("!!!!!! Error trying to save gbc model to pickle !!!!!!!:\n{}".format(ex))
            sys.exit()

        print("Finished saving data and model.")


    def run(self, images, by_state_or_type, labels_state, labels_type_str, *labels_filter, **params):
        '''
        INPUT
        images: numpy arrays representing preprocessed images from process_photos_square_padded module
        by_state_or_type: string representing what type of comparison is being done, e.g. "state" for state-by-state comparison, or "type" for mountain/mount/peak comparison
        labels_state: labels for state comparison
        labels_type_str: string labels for type comparison
        *labels_filter: labels that type comparison is based on, e.g. "CO", "UT" would compare the data for CO verus UT. "type_gbc3" or "type_gbc2" have labels generated by GradientBoostingClassifier
        **params: parameters for CNN and other functions for this class

        OUTPUT
        Nothing is returned

        Runs each batch of INPUTs through above functon in this class to generate test accuracy results, and save results in model_fit_results table in summitsdb
        '''
        #filter images/labels per labels_filter
        if by_state_or_type == 'state':
            images, labels, comparison, num_classes, classes = self.filter_images(images, labels_state, *labels_filter)

        elif by_state_or_type == 'type':
            images, labels, comparison, num_classes, classes = self.filter_images(images, labels_type_str, *labels_filter)

        elif by_state_or_type == 'type_gbc3':
            images, labels, comparison, num_classes, classes = self.filter_images(images, labels_type_gbc3, *labels_filter)

        elif by_state_or_type == 'type_gbc2':
            images, labels, comparison, num_classes, classes = self.filter_images(images, labels_type_gbc2, *labels_filter)

        else:
            print("run: by_state_or_type={}--this is an incorrect parameter.\nTERMINATING PROGRAM.".format(by_state_or_type))
            sys.exit()

        params['comparison'] = comparison
        print("params={}".format(params))

        #balance classes--do this before one hot encoding
        images, labels = self.balance_classes(images, labels, numrows_in_each_class=params['numrows_in_each_class'], num_classes=params['num_classes'], images_or_data="images")

        #convert labels to one hot encoding
        labels =self.one_hot_encode_labels(labels, num_classes)
        # print("labels[:10] after one_hot_encode_labels:\n{}".format(labels[:10]))

        #shuffle data and split data into 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20) #random_state set in np.random.seed at top
        print("#Samples:")
        print("labels images\n{}  {}".format(labels.shape[0], images.shape[0]))
        print("\nX_train y_train X_test y_test\n{}   {}   {}   {}".format(X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0]))

        #free up memory used by no longer needed images, labels lists
        images = None
        labels = None

        # print('X_train shape:', X_train.shape)
        # print('#train samples={}'.format(X_train.shape[0]))
        # print('#test samples={}'.format(X_test.shape[0]))

        print("Done preparing data and model.\nNow fitting cnn model using {} data.\n".format(image_squaring))

        # Keras input image dimensions
        print("input_shape={}".format(params['input_shape']))

        #fit cnn model on train data using **params, evaluate on test data, save results and model
        self.fit_cnn(X_train, y_train, X_test, y_test, **params)


if __name__ == "__main__":
    image_squaring = "padded"
    # cnn_image_size contains a tuple (horizontal, vertical) representing the number of pixels in the images read in later
    # labels_state contains numpy array (strings) of all states (two letter abbreviations) corresponding to and in same order as images
    # labels_type_str contain  numpy arrays (integers) equal to either "mount", "mountain", or "peak"
    with open(capstone_folder + "pickled_images_labels/cnn_image_size.pkl", 'rb') as f:
        cnn_image_size = pickle.load(f) # (length, width) of images
    print("Reading labels data...")
    with open(capstone_folder + "pickled_images_labels/labels_state.pkl", 'rb') as f:
        labels_state = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_str.pkl", 'rb') as f:
        labels_type_str = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_gbc3.pkl", 'rb') as f:
        labels_type_gbc3 = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_gbc2.pkl", 'rb') as f:
        labels_type_gbc2 = pickle.load(f) #shape=(numrows)    print("Done reading labels data.")
    print("Done reading labels data.")

    cnn_image_size = (100, 100, 3)
    print("Reading {} images data...".format(image_squaring))
    # the images files are large and take several seconds to load
    fn = "pickled_images_labels/images_"

    with open(capstone_folder + fn + image_squaring + ".pkl", 'rb') as f:
        images = pickle.load(f) #shape=(numrows, cnn_image_size[0], cnn_image_size[1], 3)

    #read data into df
    #use imported alchemy_conn program to generate sqlalchemy connection to summitsdb
    engine = alchemy_engine()

    #load summits table into pandas df
    df = pd.read_sql_query('''SELECT DISTINCT summit_id, elevation, isolation, prominence, type_str, state
    FROM summits
    WHERE type_str IN ('mount', 'mountain', 'peak')
    ORDER BY summit_id;''', con=engine)

    print("Done reading {} images data.\n".format(image_squaring))

    #runs_params is a list of parameter tuples for each model to run
    runs_params = [(5000, 'type_gbc3', ('mount', 'mountain', 'peak')), (8000, 'type_gbc2', ('mountain', 'peak')), (5000, 'state', ('CO', 'WA', 'UT')), (4000, 'state', ('NM', 'UT')), (5000, 'state', ('CO', 'WA')), (5000, 'state', ('WA', 'NM')), (5000, 'state', ('WA', 'UT')), (5000, 'state', ('CO', 'UT')), (8000, 'type', ('mountain', 'peak')), (5000, 'type', ('mount', 'mountain', 'peak'))]

    cnn_params = {'num_epochs': 12, 'batch_size': 128, 'num_filters': 32, 'pool_size': (3,3), 'kernel_size': (4,4), 'input_shape': cnn_image_size, 'dense': 128, 'dropout1': 0.25,  'dropout2': 0.25}

    # get next model# from model_fit_results table
    conn, cur = get_db_conn_cursor()
    query = 'SELECT COALESCE(MAX(model_num), 0) FROM model_fit_results'
    doquery(cur, query, "SELECT MAX model_num")
    model_num = cur.fetchone()[0] + 1
    conn.close()


    print("\n++++++++++++++ num_epochs={} ++++++++++++++\n".format(cnn_params['num_epochs']))
    print("============= Model# {} =============".format(model_num))

    for run_num, run_params in enumerate(runs_params, start=1):
        numrows_in_each_class, by_state_or_type, labels_filter = run_params

        print("************* {}. Running {} *************".format(run_num, run_params))

        params = {'comparison': '', 'type': by_state_or_type,  'numrows_in_each_class': numrows_in_each_class}
        params.update(cnn_params)
        params.update({'num_classes': len(labels_filter)})

        cnn = cnn_class(model_num)
        by_state_or_type= run_params[1]
        cnn.run(images, by_state_or_type, labels_state, labels_type_str, *labels_filter, **params)

        if 'gbc' not in by_state_or_type:
            print("Now running GradientBoosing on data.")
            cnn.run_GradientBoosting(df, by_state_or_type, numrows_in_each_class, params['comparison'], *labels_filter)

        print("************* {}. Finished {} ************\n\n".format(run_num, run_params))

    print("++++++++ FINISHED PROGRAM ++++++++")
