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
    labels: np array consisting of 2 state labels (2 letter abbrev) for images
    *choices: items in labels to filter

    OUTPUT
    np array: filtered images
    np array: filtered labels of choices, but converted to integers starting with 0 for first of choices, 1 for second, etc.
    compare_type: string showing each choice selected separated by "_"

    filter images and lables per *choices, e.g.:
    images = [1, 2, 1, 3, 1]
    labels = ['CO', 'WA', 'CO', 'NM', 'CO']
    filter_data(images, labels, 'CO', 'NM')
    returns [1, 1, 3, 1], [0, 0, 1, 0,], 'CO_NM'
    '''
    images_list = []
    labels_list = []
    compare_type = ""
    choice_int = -1
    for choice in choices:
        compare_type += choice + "_"
    compare_type = compare_type[:-1] #drop last "_" on compare_type

    for image, label in zip(images, labels):
            choice_int = -1
            for choice in choices:
                choice_int += 1
                if label == choice:
                    images_list.append(image)
                    labels_list.append(choice_int)

    return np.array(images_list), np.array(labels_list), compare_type


def fit_cnn(X_train, y_train, X_test, y_test, num_classes):
    #parameters for cnn
    batch_size = 128
    nb_classes = num_classes
    nb_epoch = 1 #12  USE 1 for TESTING purposes--faster

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

    model.add(Conv2D(filters=nb_filters, kernel_size=(kernel_size), padding='valid', input_shape=input_shape))
    # model.add(Conv2D(filters=nb_filters, kernel_size=kernel_size))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, y_test))
    # X_train.shape, y_train.shape, X_test.shape, y_test.shape

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test loss={}'.format(score[0]))
    print('Test accuracy={}'.format(score[1]))

    return model

def to_categorical_binary(y_not_categ):
    y = []
    for value in y_not_categ:
        if value == 0:
            y.append([1, 0])
        elif value == 1:
            y.append([0, 1])
        else:
            print("Error: values in y_not_categ is not 0 or 1. row= {}".format(value))

    return np.array(y)


def make_different_images(images):
    #flip image horizontally and return it
    f = lambda x: x[::-1]
    f1 = lambda x: np.apply_along_axis(f, 1, x)
    return np.array([f1(x) for x in images])


def balance_classes(X, y, toSize):
    '''
    INPUT
    X: np array of num_classes (calculated below) features
    y: np array of binary labelsof num_classes = # unique values in y
    toSize: int size resulting size of each class

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
    # toSize = 13
    # y.shape, X.shape
    # X
    # y

    uniques = np.unique(y)
    num_classes = len(uniques)
    num_per_class = np.zeros((num_classes), dtype=int)
    X_class = []
    y_class = []
    # value = 0
    # i = 0
    for i, value in enumerate(uniques):
        indices = np.where(y == value)[0]
        X_class.append(X[indices]) #converts X_class into np array
        y_class.append(np.full(toSize, value, dtype=int)) #converts y_class into np array
        num_per_class[i] += indices.size
    X_class[0].shape, y_class[0].shape
    X_class[1].shape, y_class[1].shape
    for i, num in enumerate(num_per_class):
        # i = 0   i = 1
        # num = num_per_class[i]
        if num == toSize:
            continue #already correct size, so no adjustment needed

        elif num > toSize:
        # downsample to 5000 rows
            X_class[i] = resample(X_class[i], replace = False, n_samples=toSize, random_state=1)

        else: # num < toSize
            #upsize with replacement
            #since cannot resize with replacement if n_samples > existing samples, repeat resampling several times to get to toSize
            num_iters_upsample = int(toSize / num) - 1
            num_addition_upsamples = toSize % num

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

            if num_per_class[i] < .5 * toSize:
                #split X_newsamples in 2, and apply make_different_image to half of them
                half_num_newsamples = int(len(X_newsamples) / 2)
                X_newsamples_half1 = X_newsamples[:half_num_newsamples]
                X_newsamples_half2 = X_newsamples[half_num_newsamples:]
                X_newsamples_half1.shape, X_newsamples_half2.shape

                X_newsamples_half1 = make_different_images(X_newsamples_half1)
                np.vstack((X_new, X_newsamples_half2)).shape

                X_newsamples = np.vstack((X_newsamples_half1, X_newsamples_half2))

            else: # num_per_class[i] >= .5 * toSize
                #flip X_newsamples and before appending them to X_class[i]
                X_newsamples = make_different_images(X_newsamples)

            #append new samples to existing ones
            X_class[i] = np.vstack((X_class[i], X_newsamples))


        print("X_class[{}].shape[0]={}, y_class[{}].size={} vs toSize={}".format(i, X_class[i].shape[0], i, y_class[i].size, toSize))
        #X_class[i] and y_class[i] now should have toSize samples each
    # X_class_temp = X_class
    # y_class_temp = y_class
    #
    # X_class = X_class_temp
    # y_class = y_class_temp

    #recombine the separate classes
    X = np.vstack((X_class[i] for i in range(num_classes)))
    y = np.vstack((y_class[i] for i in range(num_classes))).reshape(toSize *  num_classes)
    # y_class[0].shape, y_class[1].shape
    # X.shape, y.shape
    return X, y


if __name__ == "__main__":
    state1 = "UT"
    state2 = "WA"
    #image_squaring = 'cropped' #testing
    for image_squaring in ['cropped', 'padded']:

        #images_cropped and images_padded contain numpy arrays (squared by either cropping or padding), resized to 100x100 pixels and normalized) of all summit images ordered by image_id
        # labels contains numpy array of all states (two letter abbreviations) corresponding to and in same order as images
        print("Reading data...", end="")
        with open(capstone_folder + "images_" + image_squaring + ".pkl", 'rb') as f:
            images = pickle.load(f) #shape=(numrows, 100, 100, 3)
        with open(capstone_folder + "labels_state.pkl", 'rb') as f:
            labels = pickle.load(f) #shape=(numrows)
        print("finished.\nNow preparing data and model...", end="")
        #images.shape, labels.shape
        num_classes = 2 #UT vs WA, coded as 0 and 1
        images, labels, compare_type = filter_data(images, labels, state1, state2)
        #images.shape, labels.shape

        temp_images = images
        temp_labels = labels

        images = temp_images
        labels = temp_labels
        #balance classes--do this before converting to binary
        images, labels = balance_classes(images, labels, toSize=5000)

        #convert labels to binary form
        labels = to_categorical_binary(labels)

        #shuffle data and split data into 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=42)
        # numrows = labels.shape[0]
        # indices = np.arange(numrows)
        # np.random.seed(1337)  # for reproducibility
        # np.random.shuffle(indices)
        # num_train_indices = int(numrows * .8) #80% to train set
        # X_train = images[indices[:num_train_indices]]
        # y_train = np_utils.to_categorical(labels[indices[:num_train_indices]])
        # X_test = images[indices[num_train_indices:]]

        # y_keras = tf.keras.utils.to_categorical(y)
        # y_np = np_utils.to_categorical(y_test)
        #X_train.shape, y_train.shape, X_test.shape, y_test.shape, y_test.shape[1]
        #free up memory used by no longer needed images, labels lists
        images = None
        labels = None
        y = None

        image_rows, image_cols = 100, 100 # Keras input image dimensions
        input_shape = (image_rows, image_cols, 3)
        if K.image_dim_ordering() == 'th':
            print("\n*** K.image_dim_ordering() == 'th' ***\n")
            X_train = X_train.reshape(X_train.shape[0], 3, image_rows, image_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, image_rows, image_cols)
            input_shape = (3, image_rows, image_cols)

        # print('X_train shape:', X_train.shape)
        # print('#train samples={}'.format(X_train.shape[0]))
        # print('#test samples={}'.format(X_test.shape[0]))

        print("done.\nNow fitting model using {} data.".format(image_squaring))
        model = fit_cnn(X_train, y_train, X_test, y_test, num_classes)

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

        print()
