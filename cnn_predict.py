import numpy as np
import sys
from keras.models import model_from_json
from process_photos_np import process_image
from skimage import io
from my_libraries import *
from cnn_train import filter_data

capstone_folder, images_folder = folders()

def load_model(image_squaring):
        #read model and weight back in
        try:
            print("Reading model from json & h5...", end="")
            fn = capstone_folder + "model_" + image_squaring + "_" + compare_type
            with open(fn + ".json", "r") as f:
                loaded_model_json = f.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(fn + ".h5")
            print("done!")
        except Exception as ex:
            print("Error trying to read model fromS json & h5:\n{}".format(ex))
            sys.exit()

def predict_state(image_id, state_actual):
    model_cropped = load_model("cropped")
    model_padded = load_model("padded")

    filename = images_folder + str(image_id) + ".jpg"

    #get actual label for image
    if state_actual == 'UT':
        label_actual = 0
    elif state_actual == 'WA':
        label_actual = 1

    image = io.imread(filename)
    image_cropped, image_padded = process_image(image)
    images_cropped = np.array(list(image_cropped))
    images_padded = np.array(list(image_padded))

    images_cropped =[]
    images_padded = []

    images_cropped.append(image_cropped)
    images_padded.append(image_padded)

    #convert lists to np.arrays
    images_cropped = np.array(images_cropped)
    images_padded = np.array(images_padded)

    y_pred_cropped = model_cropped.predict(images_cropped, batch_size=None, verbose=1)
    pct_pred_cropped = y_pred_cropped[0][label_actual] * 100

    y_pred_padded = model_padded.predict(images_padded, batch_size=None, verbose=1)
    pct_pred_padded = y_pred_padded[0][label_actual] * 100

    return y_pred_cropped, y_pred_padded

if __name__ == "__main__":
    compare_type = "UT_vs_WA"
    #test on this image:
    image_id = 5
    state = 'WA' #this is what model is trying to predict
    # type_ = 1
    # type_str = "mount"

    y_pred_cropped, y_pred_padded = predict_state(image_id, state)

    print("Using cropped image, model predicted chance that state='{}'' is {:4.1f}% .".format(states[label_actual], pct_pred_cropped))

    print("Using padded image, model predicted chance that state='{}'' is {:4.1f}% .".format(states[label_actual], pct_pred_padded))
