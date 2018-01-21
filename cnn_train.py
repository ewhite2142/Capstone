import numpy as np
from keras.models import Sequential
from sklearn.utils import resample
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import pickle as pickle
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from my_libraries import *

np.RandomState = 1000 # for consistent results
capstone_folder, images_folder = folders()

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

    Filter images and lables per *choices, and convert string labels to one hot encoding, e.g.:
    images = np.array([11, 12, 13, 14, 15])
    labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
    filter_data(images, labels, 'CO', 'NM') returns:
    np.array([11, 13, 14, 15])
    np.array([0, 0, 1, 0])
    'CO_NM'
    '''
    num_classes = len(choices)
    if num_classes != len(set(choices)):
        print("*choices must contiain a unique list, and it is not unique. TERMINATING PROGRAM.")
        sys.exit()

    X = [] #images to be returned
    y = [] #converted labels to be returned
    compare_type = '' #string to be returned

    y_col_index = dict() #key=choice, value=y column index, used below
    for col_index, choice in enumerate(choices):
        y_col_index[choice] = col_index
        compare_type += choice + "_"
    compare_type = compare_type[:-1] #drop last "_" from compare_type

    for y_val, (image, label) in enumerate(zip(images, labels)):
        if label in choices:
            X.append(image)
            y.append(y_col_index[label])

    return np.array(X), np.array(y), compare_type, num_classes

# #TEST above
# images = np.array([11, 12, 13, 14, 15])
# labels = np.array(['CO', 'WA', 'CO', 'NM', 'CO'])
# X, y, compare_type = filter_data(images, labels, 'CO', 'NM')
# print ("{}\n{}\n{}".format(X, y, compare_type))

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
    batch_size = 128
    num_epochs = 1 #12  USE 1 for TESTING purposes--faster

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

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
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_data=(X_test, y_test))
    # X_train.shape, y_train.shape, X_test.shape, y_test.shape

    score = model.evaluate(X_test, y_test, verbose=0)

    print('\nTest loss={}'.format(score[0]))
    print('Test accuracy={}\n'.format(score[1]))

    return model


def make_different_images(images):
    '''flip each image in images horizontally and return them'''
    f = lambda x: x[::-1]
    f1 = lambda x: np.apply_along_axis(f, 1, x)
    return np.array([f1(x) for x in images])


def balance_classes(X, y, new_class_size, num_classes):
    '''
    INPUT
    X: np array of num_classes features
    y: np array of binary labelsof num_classes = # unique values in y
    new_class_size: int size resulting size of each class

    OUTPUT:
    X, y: np.arrays with 5000 rows
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
    uniques = np.unique(y)
    num_per_class = np.zeros((num_classes), dtype=int)
    X_class = []
    y_class = []
    # value = 0
    # i = 0
    for i, value in enumerate(uniques):
        indices = np.where(y == value)[0]
        X_class.append(X[indices]) #converts X_class into np array
        y_class.append(np.full(new_class_size, value, dtype=int)) #converts y_class into np array
        num_per_class[i] += indices.size
    X_class[0].shape, y_class[0].shape
    X_class[1].shape, y_class[1].shape
    for i, num in enumerate(num_per_class):
        # i = 0   i = 1
        # num = num_per_class[i]
        if num == new_class_size:
            continue #already correct size, so no adjustment needed

        elif num > new_class_size:
        # downsample to 5000 rows
            X_class[i] = resample(X_class[i], replace = False, n_samples=new_class_size, random_state=1)

        else: # num < new_class_size
            #upsize with replacement
            #since cannot resize with replacement if n_samples > existing samples, repeat resampling several times to get to new_class_size
            num_iters_upsample = int(new_class_size / num) - 1
            num_addition_upsamples = new_class_size % num

            # j=0  j=1
            for j in range(num_iters_upsample):
                X_newsample = resample(X_class[i], replace = False, n_samples=num, random_state=j)

                if j == 0:
                    X_newsamples = np.copy(X_newsample)
                else:
                    X_newsample, X_newsamples
                    X_newsamples = np.vstack((X_newsamples, X_newsample))
            # X_newsamples.shape

            if num_addition_upsamples > 0:
                X_newsample = resample(X_class[i], replace = False, n_samples=num_addition_upsamples, random_state=1)

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
                np.vstack((X_new, X_newsamples_half2)).shape

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
    X = np.vstack((X_class[i] for i in range(num_classes)))
    y = np.vstack((y_class[i] for i in range(num_classes))).reshape(new_class_size *  num_classes)
    # y_class[0].shape, y_class[1].shape
    # X.shape, y.shape
    return X, y


if __name__ == "__main__":
    state1 = "UT"
    state2 = "WA"
    #image_squaring = 'cropped' #testing
    for image_squaring in ['cropped', 'padded']:
        print("************* Running model with {} images *************".format(image_squaring.upper()))

        #images_cropped and images_padded contain numpy arrays (squared by either cropping or padding), resized to 100x100 pixels and normalized) of all summit images ordered by image_id
        # labels contains numpy array of all states (two letter abbreviations) corresponding to and in same order as images
        print("Reading data...", end="")
        with open(capstone_folder + "cnn_image_size.pkl", 'rb') as f:
            cnn_image_size = pickle.load(f) # (length, width) of images
        with open(capstone_folder + "images_" + image_squaring + ".pkl", 'rb') as f:
            images = pickle.load(f) #shape=(numrows, cnn_image_size[0], cnn_image_size[1], 3)
        with open(capstone_folder + "labels_state.pkl", 'rb') as f:
            labels = pickle.load(f) #shape=(numrows)

        print("done.\nNow preparing data and model...", end="")
        #images.shape, labels.shape

        images, labels, compare_type, num_classes = filter_data(images, labels, state1, state2)
        #images.shape, labels.shape

        #balance classes--do this before one hot encoding
        images, labels = balance_classes(images, labels, new_class_size=5000, num_classes=num_classes)

        #convert labels to one hot encoding
        labels = one_hot_encode_labels(labels, num_classes)

        #shuffle data and split data into 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20)

        #free up memory used by no longer needed images, labels lists
        images = None
        labels = None

        # Keras input image dimensions
        # if K.image_dim_ordering() == 'th':
        #     print("\n*** K.image_dim_ordering() == 'th' ***\n")
        #     X_train = X_train.reshape(X_train.shape[0], 3, image_rows, image_cols)
        #     X_test = X_test.reshape(X_test.shape[0], 3, image_rows, image_cols)
        #     input_shape = (3, image_rows, image_cols)

        # print('X_train shape:', X_train.shape)
        # print('#train samples={}'.format(X_train.shape[0]))
        # print('#test samples={}'.format(X_test.shape[0]))

        input_shape = (cnn_image_size[0], cnn_image_size[1], 3)
        print("done.\nNow fitting model using {} data.\n".format(image_squaring))

        model = fit_cnn(X_train, y_train, X_test, y_test, input_shape, num_classes)

        #this gives an error
        # try:
        #     print("Saving model using karis to hdf5...", end="")
        #     tf.keras.models.save_model(model=model, filepath=capstone_folder + "model_" + image_squaring + "_" + compare_type + ".hdf5", overwrite=True, include_optimizer=True)
        #     print("done!")
        # except Exception as ex:
        #     print("Error trying to use tf.keras.models.save_model\n{}".format(ex))

        #save model for future use
        try:
            print("Saving model to json & h5...", end="")
            fn = capstone_folder + "model_" + image_squaring + "_" + compare_type
            with open(fn + ".json", "w") as f:
                f.write(model.to_json())
            model.save_weights(fn + ".h5")
            print("done!")
        except Exception as ex:
            print("Error trying to save model to json & h5:\n{}".format(ex))

        print("************* Finished processing {} images *************\n".format(image_squaring))
