import numpy as np
import pickle as pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
import sys
from my_libraries import *

capstone_folder, images_folder = folders()

def create_reverse_image(image):
    return image[:,::-1]

def filter_data(images, labels, compare_type):
    if compare_type == "UT_vs_WA":
        ### see if cnn can label whether summit is in UT or WA ###
        #gather images & labels from labels of UT and WA: label UT=0, WA=1
        images_list = []
        labels_list = []
        for image, state in zip(images, labels):
            if state == 'UT':
                images_list.append(image)
                labels_list.append(0)
            elif state == 'WA':
                images_list.append(image)
                labels_list.append(1)

        return np.array(images_list), np.array(labels_list)

    else:
        print("Program not written for compare_type={}.\n**** EXITING PROGRAM ****".format(compare_type))
        sys.exit()


def fit_cnn(X_train, y_train, X_test, y_test, number_classes):
    #parameters for cnn
    batch_size = 128
    nb_classes = number_classes
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

#image_squaring = 'cropped' #testing
for image_squaring in ['cropped', 'padded']:

    #images_cropped and images_padded contain numpy arrays (squared by either cropping or padding), resized to 100x100 pixels and normalized) of all summit images ordered by image_id
    # labels contains numpy array of all states (two letter abbreviations) corresponding to and in same order as images
    compare_type = "UT_vs_WA"
    print("Reading data...", end="")
    with open(capstone_folder + "images_" + image_squaring + ".pkl", 'rb') as f:
        images = pickle.load(f) #shape=(numrows, 100, 100, 3)
    with open(capstone_folder + "labels_state.pkl", 'rb') as f:
        labels = pickle.load(f) #shape=(numrows)
    print("finished.\nNow preparing data and model...", end="")
    #images.shape, labels.shape
    number_classes = 2 #UT vs WA, coded as 0 and 1
    images, labels = filter_data(images, labels, compare_type)
    #images.shape, labels.shape

    #shuffle data and split data into 80/20 train/test split
    numrows = labels.shape[0]
    indices = np.arange(numrows)
    np.random.seed(1337)  # for reproducibility
    np.random.shuffle(indices)
    num_train_indices = int(numrows * .8) #80% to train set
    X_train = images[indices[:num_train_indices]]
    y_train = np_utils.to_categorical(labels[indices[:num_train_indices]])
    X_test = images[indices[num_train_indices:]]
    y_test = np_utils.to_categorical(labels[indices[num_train_indices:]])
    #np.unique(y_train), np.unique(y_test)
    #X_train.shape, y_train.shape, X_test.shape, y_test.shape
    #free up memory used by no longer needed images, labels lists
    images = None
    labels = None

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
    model = fit_cnn(X_train, y_train, X_test, y_test, number_classes)

    try:
        print("Saving model using karis to hdf5...", end="")
        tf.keras.models.save_model(model=model, filepath=capstone_folder + "model_" + image_squaring + "_" + compare_type + ".hdf5", overwrite=True, include_optimizer=True)
        print("done!")
    except Exception as ex:
        print("Error trying to use tf.keras.models.save_model\n{}".format(ex))

    try:
        print("Saving model to yaml & h5...", end="")
        fn = capstone_folder + "model_" + image_squaring + "_" + compare_type
        with open(fn + ".yaml", "w") as f:
            f.write(model.to_yaml())
        model.save_weights(fn + "h5")
        print("done!")
    except Exception as ex:
        print("Error trying to save model to yaml & h5:\n{}".format(ex))

    print()

    #with open(capstone_folder + "model_" + image_squaring + "_" + compare_type + ".pkl", "wb") as f:
        # pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
