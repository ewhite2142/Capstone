import numpy as np
np.random.seed(1000) #to get consistent results every time--used to test hyperparameter

from sklearn.utils import resample
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import sys
import pickle as pickle
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from my_libraries import *
from tensorflow import set_random_seed
from keras.models import Sequential
set_random_seed(1000)

capstone_folder, images_folder = folders()
# np.random.RandomState = 1000 #does NOT give consistent results every time--use np.random.seed instead

def filter_data(images, labels, *choices):
    '''
    INPUT
    images: np array of images
    labels: np array of strings--e.g. either state labels (2 letter abbrev), or "mount"/"mountain"/"peak" for summit type
    *choices: strings --> unique subset of labels to filter images and labels by

    OUTPUT
    np array: subset of images corresponding to labels filtered by *choices
    np array: subset of labels filtered by *choices, but contered to integers starting with 0 for first item in choices
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


def balance_classes(X, y, new_class_size, num_classes):
    '''
    INPUT
    X: np array of num_classes features
    y: np array of binary labelsof num_classes = # unique values in y
    new_class_size: int size resulting size of each class

    OUTPUT:
    X, y: np.arrays with new_class_size rows
    '''
    # X = images #TESTING
    # y = labels #TESTING
    # images.shape, labels.shape
    # if 1 == 1:
    #     print("balance_classes: STOPPING EARLY for testing.")
    #     sys.exit() # TESTING
    # X = np.arange(21).reshape(7,1,1,3)
    # y = np.array([0,1,2,0,0,0,1])
    # new_class_size = 13
    # y.shape, X.shape
    # X
    # y
    # print("\nbalance_classes--before: X.shape {}, y.shape {}".format(X.shape, y.shape))
    classes = np.unique(y)
    # print("classes: {}".format(classes))
    num_per_class = np.zeros((num_classes), dtype=int)
    X_class = [] #each element of list-->features for that class
    y_class = [] #each element of list-->labels for that class
    # value = 0
    # i = 0
    for i, value in enumerate(classes):
        indices = np.where(y == value)[0]
        X_class.append(X[indices]) # features for this class
        y_class.append(np.full(new_class_size, value, dtype=int)) # labels for this class with num_per_class rows
        num_per_class[i] = indices.size
    # X_class[0].shape, y_class[0].shape
    # X_class[1].shape, y_class[1].shape

    #for each class in X, make that class have new_class_size number of rows
    for i, num in enumerate(num_per_class):
        # i = 0   i = 1
        # num = num_per_class[i]
        if num == new_class_size:
            continue #already correct size, so no adjustment needed

        elif num > new_class_size:
        # downsample to new_class_size rows
            X_class[i] = resample(X_class[i], replace = False, n_samples=new_class_size) #random_state set in np.random.seed at top

        else: # num < new_class_size
            #upsize with replacement
            #since cannot resize with replacement if n_samples > existing samples, repeat resampling several times to get to new_class_size
            num_iters_upsample = int(new_class_size / num) - 1
            num_addition_upsamples = new_class_size % num

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

            if num_per_class[i] < .5 * new_class_size:
                #split X_newsamples in 2, and apply make_different_image to half of them
                half_num_newsamples = int(len(X_newsamples) / 2)
                X_newsamples_half1 = X_newsamples[:half_num_newsamples]
                X_newsamples_half2 = X_newsamples[half_num_newsamples:]
                X_newsamples_half1.shape, X_newsamples_half2.shape

                X_newsamples_half1 = make_different_images(X_newsamples_half1)
                np.vstack((X_newsamples_half1, X_newsamples_half2))

                X_newsamples = np.vstack((X_newsamples_half1, X_newsamples_half2))

            else: # num_per_class[i] >= .5 * new_class_size
                #flip X_newsamples and before appending them to X_class[i]
                X_newsamples = make_different_images(X_newsamples)

            #append new samples to existing ones
            X_class[i] = np.vstack((X_class[i], X_newsamples))


        # print("X_class[{}].shape[0]={}, y_class[{}].size={} vs new_class_size={}".format(i, X_class[i].shape[0], i, y_class[i].size, new_class_size))
        #X_class[i] and y_class[i] now should have new_class_size samples each
    # X_class_temp = X_class
    # y_class_temp = y_class
    #
    # X_class = X_class_temp
    # y_class = y_class_temp

    #recombine the separate classes
    # print("balance_classes: y_class[0].shape={}, y_class[1].shape={}".format(y_class[0].shape, y_class[1].shape))

    X = np.vstack((X_class[i] for i in range(num_classes)))
    y = np.vstack((np.array(y_class[i]) for i in range(num_classes))).reshape(new_class_size *  num_classes)
    print()
    # y_class[0].shape, y_class[1].shape
    # X.shape, y.shape
    # print("balance_classes--after: X.shape {}, y.shape {}".format(X.shape, y.shape))
    return X, y


def make_different_images(images):
    '''flip each image in images horizontally and return them'''
    f = lambda x: x[::-1]
    f1 = lambda x: np.apply_along_axis(f, 1, x)
    return np.array([f1(x) for x in images])


def one_hot_encode_labels(y_not_categ, num_classes):
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


def fit_cnn(X_train, y_train, X_test, y_test, input_shape, num_classes):
    #parameters for cnn
    batch_size = 150 #128
    num_epochs = 12 #12  USE 1 for TESTING purposes--faster

    # number of convolutional filters to use
    nb_filters = 30 #32
    # size of pooling area for max pooling
    pool_size = (3,3) #(2, 2)
    # convolution kernel size
    kernel_size = (4,4) #(3, 3)

    #create Sequential cnn in Keras
    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Conv2D(filters=nb_filters, kernel_size=(kernel_size), padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))

    # model.add(Conv2D(filters=nb_filters, kernel_size=(kernel_size), padding='valid', input_shape=input_shape))
    model.add(Conv2D(filters=nb_filters, kernel_size=kernel_size))

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

    return model, score[1] # model, accuracy

def run(images, labels, compare_type, num_classes, classes, class_size):
    #balance classes--do this before one hot encoding
    images, labels = balance_classes(images, labels, new_class_size=class_size, num_classes=num_classes)

    #convert labels to one hot encoding
    labels = one_hot_encode_labels(labels, num_classes)

    #shuffle data and split data into 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20) #random_state set in np.random.seed at top
    print("#Samples:")
    print("labels images\n{}  {}".format(labels.shape[0], images.shape[0]))
    print("\nX_train y_train X_test y_test\n{}   {}   {}   {}".format(X_train.shape[0], y_train.shape[0], X_test.shape[0], y_test.shape[0]))

    #free up memory used by no longer needed images, labels lists
    images = None
    labels = None

    # if K.image_dim_ordering() == 'th':
    #     print("\n*** K.image_dim_ordering() == 'th' ***\n")
    #     X_train = X_train.reshape(X_train.shape[0], 3, image_rows, image_cols)
    #     X_test = X_test.reshape(X_test.shape[0], 3, image_rows, image_cols)
    #     input_shape = (3, image_rows, image_cols)

    # print('X_train shape:', X_train.shape)
    # print('#train samples={}'.format(X_train.shape[0]))
    # print('#test samples={}'.format(X_test.shape[0]))

    print("Done preparing data and model.\nNow fitting model using {} data.\n".format(image_squaring))

    # Keras input image dimensions
    input_shape = (cnn_image_size[0], cnn_image_size[1], 3)
    print("input_shape={}".format(input_shape))
    #fit model
    model, accuracy = fit_cnn(X_train, y_train, X_test, y_test, input_shape, num_classes)

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
    print("************* Finished processing {} images *************\n".format(image_squaring))

    return accuracy



if __name__ == "__main__":
    image_squaring = "padded"
    print("Reading labels data...")
    # cnn_image_size contains a tuple (horizontal, vertical) representing the number of pixels in the images read in later
    # labels_state contains numpy array (strings) of all states (two letter abbreviations) corresponding to and in same order as images
    # labels_type_str contain  numpy arrays (integers) equal to either "mount", "mountain", or "peak"
    # with open(capstone_folder + "pickled_images_labels/cnn_image_size.pkl", 'rb') as f:
    #     cnn_image_size = pickle.load(f) # (length, width) of images
    cnn_image_size = (100, 100)
    print("Reading {} images data...".format(image_squaring))
    # the images files are large and take several seconds to load
    fn = "pickled_images_labels/images_"
    with open(capstone_folder + fn + image_squaring + ".pkl", 'rb') as f:
        images = pickle.load(f) #shape=(numrows, cnn_image_size[0], cnn_image_size[1], 3)
    print("Done reading {} images data.\nNow preparing data and model...".format(image_squaring))

    #image_squaring = 'cropped' #testing
    print("************* Running model with {} images *************".format(image_squaring.upper()))

    # images_cropped and images_padded contain numpy arrays of all summit images ordered by image_id, made to be all the same size by either cropping or padding, downsized (e.g. to 100x100 pixels--the size is stored in cnn_image_size.pkl which is read above) and normalized.
    results = dict()
    for i in range(7):
        if i == 0:
            with open(capstone_folder + "pickled_images_labels/labels_state.pkl", 'rb') as f:
                labels_state = pickle.load(f) #shape=(numrows)

            images, labels, compare_type, num_classes, classes = filter_data(images, labels_state, "NM", "UT")
            class_size = 4000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 1:
            images, labels, compare_type, num_classes, classes = filter_data(images, labels_state, "CO", "WA")
            class_size = 5000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 2:
            images, labels, compare_type, num_classes, classes = filter_data(images, labels_state, "WA", "NM")
            class_size = 5000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 3:
            images, labels, compare_type, num_classes, classes = filter_data(images, labels_state, "WA", "UT")
            class_size = 5000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 4:
            images, labels, compare_type, num_classes, classes = filter_data(images, labels_state, "CO", "UT")
            class_size = 5000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 5:
            labels_state = None # to free up memory

            with open(capstone_folder + "pickled_images_labels/labels_type_str.pkl", 'rb') as f:
                labels_type_str = pickle.load(f) #shape=(numrows)

            images, labels, compare_type, num_classes, classes = filter_data(images, labels_type_str, "mountain", "peak")
            class_size = 8000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

        if i == 6:
            images, labels, compare_type, num_classes, classes = filter_data(images, labels_type_str, "mount", "mountain", "peak")
            class_size = 5000
            accuracy = run(images, labels, compare_type, num_classes, classes, class_size)
            results[(compare_type, class_size)] = accuracy

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
