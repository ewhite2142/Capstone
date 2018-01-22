import numpy as np
import sys
from keras.models import model_from_json
from process_photos_square_cropped_padded import process_image
from skimage import io
from my_libraries import *
from cnn_train import filter_data

capstone_folder, images_folder = folders()

def load_model(image_squaring):
    # image_squaring="padded"
    # compare_type = "UT_WA"
        #read model and weight back in
    try:
        print("Reading model from json & h5...")
        fn = capstone_folder + "models/model_" + image_squaring + "_" + compare_type
        with open(fn + ".json", "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(fn + ".h5")
        print("done readng model.")
    except Exception as ex:
        print("Error trying to read model from json & h5:\n{}".format(ex))
        sys.exit()

    return model

def predict_state(image_id, state_actual, ):
    # model_cropped = load_model("cropped")
    model_padded = load_model("padded")

    filename = images_folder + str(image_id) + ".jpg"

    image = io.imread(filename)
    image_cropped, image_padded = process_image(image)
    images_padded = np.array(list(image_padded))

    images_padded = []

    # images_cropped.append(image_cropped)
    images_padded.append(image_padded)

    #convert lists to np.arrays
    images_padded = np.array(images_padded)

    y_pred_padded = model_padded.predict(images_padded, batch_size=None, verbose=1)

    return y_pred_padded, pct_pred_padded

if __name__ == "__main__":
    compare_type = "UT_WA"
    #test on this image:
    image_id = 5
    state = 'WA' #this is what model is trying to predict
    # type_ = 1
    # type_str = "mount"

    y_pred_padded, pct_pred_padded = predict_state(image_id, state)

    # print("Using cropped image, model predicted chance that state='{}'' is {:4.1f}% .".format(y_pred_cropped[label_actual], pct_pred_cropped))

    print("Using padded image, model predicted chance that state='{}'' is {:4.1f}% .".format(y_pred_padded[1], pct_pred_padded))
