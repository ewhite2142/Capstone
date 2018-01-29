import numpy as np
import pandas as pd
import sys
from keras.models import model_from_json
from process_photos_square_cropped_padded import process_image
from skimage import io
import matplotlib.pyplot as plt
import pickle as pickle
from my_libraries import *
from alchemy_conn import alchemy_engine

capstone_folder, images_folder = folders()

#use imported alchemy_conn program to generate sqlalchemy connection to summitsdb
engine = alchemy_engine()

#load summits table into pandas df
df = pd.read_sql_query('''SELECT DISTINCT summit_id, name, elevation, isolation, prominence, type, type_str, state
FROM summits
WHERE summit_id IN (12, 17, 16686, 12401)
''', con=engine)

comparisons = ['mountain_peak_by_type', 'mount_mountain_peak_by_type', 'CO_UT_by_state', 'WA_NM_by_state']

images_fn = ['Grays Peak CO.jpg', 'Mt. Evans CO.jpg', 'Mount Peale Utah.jpg', 'Animas Peak NM.jpg']

summit_ids = [12, 17, 16686, 12401]

y_values = [1, 0, 1, 1]
# comparison='mountain_peak_by_type'  image_fn='Grays Peak CO.jpg'
# summit_id=12   y_value=1
for comparison, image_fn, summit_id, y_value in zip(comparisons, images_fn, summit_ids, y_values):
    model_filename = capstone_folder + "models/cnn_model_1_" + comparison
    with open(model_filename + ".json", "r") as f:
        loaded_model_json = f.read()
    cnn_model = model_from_json(loaded_model_json)
    cnn_model.load_weights(model_filename + ".h5")

    model_filename = capstone_folder + "models/gbc_model_1_" + comparison
    with open(model_filename + ".pkl", 'rb') as f:
        gbc_model = pickle.load(f)

    model_filename = capstone_folder + "models/minmax_1_" + comparison
    with open(model_filename + ".pkl", 'rb') as f:
        X_min, X_max = pickle.load(f)

    image_filename = capstone_folder + "test_cnn_images/" + image_fn

    image = io.imread(image_filename)
    plt.imshow(image)
    # plt.show()
    image_cropped, image_padded = process_image(image)
    images_padded = np.array(list(image_padded))

    images_padded = []
    images_padded.append(image_padded)
    images_padded = np.array(images_padded)

    y_pred_cnn = 100 * cnn_model.predict(images_padded, batch_size=None, verbose=0, steps=None)

    print("y_pred_cnn={:5.1f}%".format(y_pred_cnn[0][y_value]))
    print("y_pred_cnn={}".format(y_pred_cnn))

    X = df[(df['summit_id'] == summit_id)][['elevation', 'isolation', 'prominence']].values
    # print("type(X)={}\nX={}".format(type(X), X))

    # y.append(y_value)


    y = np.array(y_value)
    # print("X={}, y={}\nX_min={}, X_max={}".format(X, y, X_min, X_max))

    #normalize features data to range 0 - 1
    X = (X - X_min) / (X_max - X_min)

    y_pred_gbc = gbc_model.predict(X)
    y_pred_gbc_prob = gbc_model.predict_proba(X)
    print("y_pred_gbc={}, prob={}".format(y_pred_gbc[0], y_pred_gbc_prob[0][y_value]))
    print("++++++++++++++++++++++++++++++++++")
