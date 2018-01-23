import numpy as np
np.random.seed(1000) #to get consistent results every time--used to test hyperparameter

from sklearn.utils import resample
import sys
import pickle as pickle
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from my_libraries import *
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
import os

set_random_seed(1000)

home = os.getenv("HOME")
capstone_folder, images_folder = folders()
# np.random.RandomState = 1000 #does NOT give consistent results every time--use np.random.seed instead

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='ed', password=p)
    p = None


class cnn_class(object):
    '''
    Generates and saves in json files fitted cnn models on images (np arrays) dataset previously prepared (made into standard sqaure size, and downsized to 100x100 pixels) by a process_photos module and saved

    Each image input and processed in this module has shape(rows of images, 100, 100, 3). 100x100x3 is an image of numrows horizontal, numcols vertical, and 3 colors RGB--this is the tensorflow input_shape, as opposed to the theano shape of 3x100x100, which is not used here.
    '''
    def __init__(self):
        pass

    def filter_data(self, images, labels, *choices):
        '''
        INPUT
        images: np array of images
        labels: np array of strings--e.g. either state labels (2 letter abbrev), or "mount"/"mountain"/"peak" for summit type
        *choices: strings --> unique subset of labels to filter images and labels by

        OUTPUT
        np array: subset of images corresponding to labels filtered by *choices
        np array: subset of labels filtered by *choices, but convertered to integers starting with 0 for first item in choices
        string: concatenation of each of *choices separated by "_"
        integer: len(*choices)
        list--strings naming each class, sorted alphabetically

        Filter images and lables per *choices, and convert string labels to one hot encoding, e.g.:
        images = np.array([11, 12, 13, 14, 15])
        labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
        filter_data(images, labels, 'CO', 'NM') returns:
        np.array([11, 13, 14, 15])
        np.array([0, 0, 1, 0])
        'CO_NM'
        2
        ['CO', 'NM']
        '''
        num_classes = len(choices)

        if num_classes != len(set(choices)):
            print("*choices must contiain a unique list, and it is not unique. TERMINATING PROGRAM.")
            sys.exit()
        classes = sorted(list(set(choices))) #list of unique classes to be returned

        X = [] #images to be returned
        y = [] #converted labels to be returned
        compare_type = '' #string to be returned
        classes = [] #list to be returned
        y_col_index = dict() #key=choice, value=y column index, used below

        print("Classes being compared:")
        for col_index, choice in enumerate(choices):
            print(col_index, choice)
            if choice not in labels:
                print("/n/nchoice: {} is not in labels./nMAKE SURE YOU'VE SELECTED THE CORRECT choices FOR THESE labels.\nTERMINATING PROGRAM.".format(choice))
                sys.exit()

            y_col_index[choice] = col_index
            compare_type += choice + "_"
            classes.append(choice)
        compare_type = compare_type[:-1] #drop last "_" from compare_type

        for y_val, (image, label) in enumerate(zip(images, labels)):
            if label in choices:
                X.append(image)
                y.append(y_col_index[label])

        return np.array(X), np.array(y), compare_type, num_classes, classes

    # #TEST above
    # images = np.array([11, 12, 13, 14, 15])
    # labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
    # X, y, compare_type, num_classes, classes = filter_data(images, labels, 'WA', 'CO')
    # print ("{}\n{}\n{}\n{}\n{}".format(X, y, compare_type, num_classes, classes))


    def balance_classes(self, X, y, numrows_in_each_class, num_classes):
        '''
        INPUT
        X: np array--images
        y: np array--integer labels (not yet one hot encoded)
        num_classes int--# unique values in y
        numrows_in_each_class: int--desired size of each class

        OUTPUT:
        X, y: np.arrays with numrows_in_each_class number of rows for each class

        Makes X, y have same number of rows (INPUT num_classes) for each class in y. To extent upsampling is required, some of the new images added are flipped horizontally.
        '''
        # X = images #TESTING
        # y = labels #TESTING
        # images.shape, labels.shape
        # if 1 == 1:
        #     print("balance_classes: STOPPING EARLY for testing.")
        #     sys.exit() # TESTING
        # X = np.arange(21).reshape(7,1,1,3)
        # y = np.array([0,1,2,0,0,0,1])
        # numrows_in_each_class = 13
        # y.shape, X.shape
        # X
        # y
        # print("\nbalance_classes--before: X.shape {}, y.shape {}".format(X.shape, y.shape))
        classes = np.unique(y)
        # print("classes: {}".format(classes))
        num_per_class = np.zeros((num_classes), dtype=int)
        X_class = [] #each element of list = features for that class
        y_class = [] #each element of list = labels for that class
        # value = 0
        # i = 0
        for i, value in enumerate(classes):
            indices = np.where(y == value)[0]
            X_class.append(X[indices]) # features for this class
            y_class.append(np.full(numrows_in_each_class, value, dtype=int)) # labels for this class i. Goal is to make each X_class[i] and y_class[i] have num_classes rows each, with some of the new X_class[i] images that may be added flipped horizontally. Each y_class[i] has same labels, so order of rows within each X_class[i] doesn't matter. Each X_class[i] and y_class[i] are reassembled at end of this function in order of i's to maintain feature-label consistency.
            num_per_class[i] = indices.size
            # X_class[0].shape, y_class[0].shape
            # X_class[1].shape, y_class[1].shape

        #for each class in X, make that class have numrows_in_each_class number of rows
        for i, num in enumerate(num_per_class):
            # i = 0   i = 1
            # num = num_per_class[i]
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

                # j=0  j=1
                for j in range(num_iters_upsample):
                    X_newsample = resample(X_class[i], replace = False, n_samples=num) #random_state set in np.random.seed at top

                    if j == 0:
                        X_newsamples = np.copy(X_newsample)
                    else:
                        X_newsample, X_newsamples
                        X_newsamples = np.vstack((X_newsamples, X_newsample))
                # X_newsamples.shape

                if num_addition_upsamples > 0:
                    X_newsample = resample(X_class[i], replace = False, n_samples=num_addition_upsamples) #random_state set in np.random.seed at top

                    if num_iters_upsample == 0:
                        X_newsamples = np.copy(X_newsample)
                    else:
                        X_newsamples = np.vstack((X_newsamples, X_newsample))

                    # X_newsample.shape, X_newsamples.shape
                    # X_newsamples = np.append(X_newsamples, X_newsample)
                # X_newsamples.shape

                if num_per_class[i] < .5 * numrows_in_each_class:
                    #split X_newsamples in 2, and apply make_different_image to half of them
                    half_num_newsamples = int(len(X_newsamples) / 2)
                    X_newsamples_half1 = X_newsamples[:half_num_newsamples]
                    X_newsamples_half2 = X_newsamples[half_num_newsamples:]
                    X_newsamples_half1.shape, X_newsamples_half2.shape

                    X_newsamples_half1 = self.make_different_images(X_newsamples_half1)
                    np.vstack((X_newsamples_half1, X_newsamples_half2))

                    X_newsamples = np.vstack((X_newsamples_half1, X_newsamples_half2))

                else: # num_per_class[i] >= .5 * numrows_in_each_class
                    #flip X_newsamples and before appending them to X_class[i]
                    X_newsamples = self.make_different_images(X_newsamples)

                #append new samples to existing ones
                X_class[i] = np.vstack((X_class[i], X_newsamples))


            # print("X_class[{}].shape[0]={}, y_class[{}].size={} vs numrows_in_each_class={}".format(i, X_class[i].shape[0], i, y_class[i].size, numrows_in_each_class))
            #X_class[i] and y_class[i] now should have numrows_in_each_class samples each
        # X_class_temp = X_class
        # y_class_temp = y_class
        #
        # X_class = X_class_temp
        # y_class = y_class_temp

        #recombine the separate classes
        # print("balance_classes: y_class[0].shape={}, y_class[1].shape={}".format(y_class[0].shape, y_class[1].shape))

        X = np.vstack((X_class[i] for i in range(num_classes)))
        y = np.vstack((np.array(y_class[i]) for i in range(num_classes))).reshape(numrows_in_each_class *  num_classes)
        print()
        # y_class[0].shape, y_class[1].shape
        # X.shape, y.shape
        # print("balance_classes--after: X.shape {}, y.shape {}".format(X.shape, y.shape))
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
        INPUT: np array of integer labels
        OUTPUT: np array of one hot encoded labels

        Turn integer labels into one hot encoded labels, e.g.
        one_hot_encode_labels(np.array([0, 0, 1, 0]), 2) returns:
        np.array([1, 0], [1, 0], [0, 1], [1, 0])
        '''
        # y_not_categ = np.array([0, 0, 1, 0])
        # num_classes = 2
        y = np.zeros((len(y_not_categ), len(np.unique(y_not_categ))))
        for y_row, y_val in enumerate(y_not_categ):
            y[y_row, y_val] = 1
        return np.array(y)

    # #TEST above
    # print(one_hot_encode_labels(np.array([0, 0, 1, 0]), 2))


    def fit_cnn(self, X_train, y_train, X_test, y_test, input_shape, num_classes, num_epochs):
        '''
        INPUT
        X_train, y_train, X_test, y_test: np.arrays of X images, y labels (one hot vectors) split into train, test sets
        input_shape: tuple of integers (length, width)
        num_classes: int representing number of unique labels

        OUTPUT
        cnn_model: fit with INPUT X_train, y_train
        float: test_accuracy of of fitted model on X_test, y_test
        '''
        #parameters for cnn
        batch_size = 150 #128

        # number of convolutional filters to use
        num_filters = 30 #32
        # size of pooling area for max pooling
        pool_size = (3,3) #(2, 2)
        # convolution kernel size
        kernel_size = (4,4) #(3, 3)

        #create Sequential cnn in Keras
        model = Sequential()
        # 2 convolutional layers followed by a pooling layer followed by dropout
        model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=num_filters, kernel_size=kernel_size))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        # transition to an mlp
        model.add(Flatten())
        model.add(Dense(128)) #128 #400
        model.add(Activation('relu'))
        model.add(Dropout(0.25)) #0.5
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam', #adadelta
                      metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                  verbose=1, validation_data=(X_test, y_test))
        # X_train.shape, y_train.shape, X_test.shape, y_test.shape

        score = model.evaluate(X_test, y_test, verbose=0)
        print("model.metrics={}, score={}".format(model.metrics, score))

        # print('\nTest loss={}'.format(score[0]))
        print('Test accuracy={} <-------------------------\n'.format(score[1]))

        return model, score[1] # model, test_accuracy


    def run(self, images, labels_state, labels_type_str, numrows_in_each_class, by_state_or_type, num_epochs, *labels_filter):
        '''
        Runs each batch of INPUTs through steps to generate test accuracy results, and save results in a file
        '''
        results = dict() #stores model type: test_accuracy results of each model run

        if by_state_or_type == 'by_state':
            images, labels, compare_type, num_classes, classes = self.filter_data(images, labels_state, *labels_filter)

        elif by_state_or_type == 'by_type':
            images, labels, compare_type, num_classes, classes = self.filter_data(images, labels_type_str, *labels_filter)

        elif by_state_or_type == 'by_type_GBC3':
            images, labels, compare_type, num_classes, classes = self.filter_data(images, labels_type_GBC3, *labels_filter)

        elif by_state_or_type == 'by_type_GBC2':
            images, labels, compare_type, num_classes, classes = self.filter_data(images, labels_type_GBC2, *labels_filter)

        else:
            print("run: by_state_or_type={}--this is an incorrect parameter.\nTERMINATING PROGRAM.".format(by_state_or_type))
            sys.exit()

        #balance classes--do this before one hot encoding
        images, labels = self.balance_classes(images, labels, numrows_in_each_class=numrows_in_each_class, num_classes=num_classes)

        #convert labels to one hot encoding
        labels =self.one_hot_encode_labels(labels, num_classes)

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

        print("Done preparing data and model.\nNow fitting model using {} data.\n".format(image_squaring))

        # Keras input image dimensions
        input_shape = (cnn_image_size[0], cnn_image_size[1], 3)
        print("input_shape={}".format(input_shape))
        #fit model
        model, test_accuracy = self.fit_cnn(X_train, y_train, X_test, y_test, input_shape, num_classes, num_epochs)

        results[(compare_type, numrows_in_each_class)] = test_accuracy

        #this gives an error
        # try:
        #     print("Saving model using karis to hdf5...")
        #     tf.keras.models.save_model(model=model, filepath=capstone_folder + "model_" + image_squaring + "_" + compare_type + ".hdf5", overwrite=True, include_optimizer=True)
        #     print("Done savming model using karis!")
        # except Exception as ex:
        #     print("Error trying to use tf.keras.models.save_model\n{}".format(ex))

        #save model for future use
        try:
            print("Saving model to json & h5...")
            fn = capstone_folder + "models/model_" + image_squaring + "_" + compare_type
            with open(fn + ".json", "w") as f:
                f.write(model.to_json())
            model.save_weights(fn + ".h5")
            print("Done saving model to json & h5.")
        except Exception as ex:
            print("!!!!!! Error trying to save model to json & h5 !!!!!!!:\n{}".format(ex))
            sys.exit()

        #save results to DB
        #I learned the hard way...This psycopg2 function has to be within the same class and module as it is being run it; otherwise, results from SQL queries will be non-existant even with no exception thrown.
        def doquery(curs, query, title):
            #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
            try:
                curs.execute(query)
                cur.connection.commit()
            except Exception as ex:
                print('\n********************************************* error in {}\n{}'.format(title, ex))
                cur.connection.close()
                sys.exit()

        cur = conn.cursor()

        #delete row if already exists
        # compare_type="NM_UT"
        # numrows_in_each_class=2
        # num_epochs=1
        # test_accuracy = 0.54897898
        query = '''
        SELECT EXISTS (SELECT * FROM model_fit_results
        WHERE compare_type='{}' AND numrows_in_each_class={} and num_epochs={});
        '''.format(compare_type, numrows_in_each_class, num_epochs)
        doquery(cur, query, "EXISTS row in model_fit_results")
        already_in_db = cur.fetchone()[0]
        if already_in_db:
            query = '''
            DELETE FROM model_fit_results
            WHERE compare_type='{}' AND numrows_in_each_class={} and num_epochs={};
            '''.format(compare_type, numrows_in_each_class, num_epochs)
            doquery(cur, query, "DELETE existing row in model_fit_results")
            print("already in db: {}".format(query))

        #insert results into model_fit_results table
        query = '''
        INSERT INTO model_fit_results
        (compare_type, numrows_in_each_class, num_epochs, test_accuracy) VALUES
        ('{}', {}, {}, {});
        '''.format(compare_type, numrows_in_each_class, num_epochs, test_accuracy*100)
        doquery(cur, query, "INSERT INTO model_fit_results")

        cur.close()

        return results



if __name__ == "__main__":
    image_squaring = "padded"
    # cnn_image_size contains a tuple (horizontal, vertical) representing the number of pixels in the images read in later
    # labels_state contains numpy array (strings) of all states (two letter abbreviations) corresponding to and in same order as images
    # labels_type_str contain  numpy arrays (integers) equal to either "mount", "mountain", or "peak"
    # with open(capstone_folder + "pickled_images_labels/cnn_image_size.pkl", 'rb') as f:
    #     cnn_image_size = pickle.load(f) # (length, width) of images
    print("Reading labels data...")
    with open(capstone_folder + "pickled_images_labels/labels_state.pkl", 'rb') as f:
        labels_state = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_str.pkl", 'rb') as f:
        labels_type_str = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_GBC3.pkl", 'rb') as f:
        labels_type_GBC3 = pickle.load(f) #shape=(numrows)

    with open(capstone_folder + "pickled_images_labels/labels_type_GBC2.pkl", 'rb') as f:
        labels_type_GBC2 = pickle.load(f) #shape=(numrows)    print("Done reading labels data.")
    print("Done reading lables data.")

    cnn_image_size = (100, 100)
    print("Reading {} images data...".format(image_squaring))
    # the images files are large and take several seconds to load
    fn = "pickled_images_labels/images_"
    with open(capstone_folder + fn + image_squaring + ".pkl", 'rb') as f:
        images = pickle.load(f) #shape=(numrows, cnn_image_size[0], cnn_image_size[1], 3)
    print("Done reading {} images data.\n".format(image_squaring))

    #runs is a list of parameter tuples for each model to run
    runs = [(5000, 'by_type_GBC3', ('mount', 'mountain', 'peak')), (8000, 'by_type_GBC2', ('mountain', 'peak')), (5000, 'by_state', ('CO', 'WA', 'UT')), (4000, 'by_state', ('NM', 'UT')), (5000, 'by_state', ('CO', 'WA')), (5000, 'by_state', ('WA', 'NM')), (5000, 'by_state', ('WA', 'UT')), (5000, 'by_state', ('CO', 'UT')), (8000, 'by_type', ('mountain', 'peak')), (5000, 'by_type', ('mount', 'mountain', 'peak'))]
    num_epochs = 12
    print("\n++++++++++++++ num_epochs={} ++++++++++++++\n".format(num_epochs))
    for run_num, run_ in enumerate(runs, start=1):
        # run_=runs[5]  run_num=5  #TESTING
        # if run_num != 4: continue #TESTING
        numrows_in_each_class, by_state_or_type, labels_filter = run_
        print("************* {}. Running {} *************".format(run_num, run_))

        cnn = cnn_class()
        results = cnn.run(images, labels_state, labels_type_str, numrows_in_each_class, by_state_or_type, num_epochs, *labels_filter)

        print("************* {}. Finished {} ************\n\n".format(run_num, run_))
    try:
        print("Saving results...")
        fn = capstone_folder + "models/results"
        with open(fn + ".pkl", "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done saving results.  FINISHED!!!!!!!!!")
    except Exception as ex:
        print("!!!!!! Error trying to save results !!!!!!!:\n{}".format(ex))
        sys.exit()

    print("++++++++ FINISHED PROGRAM ++++++++")
