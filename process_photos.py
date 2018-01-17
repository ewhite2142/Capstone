import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import psycopg2
import sys
import itertools
import _pickle as pickle
from skimage.transform import resize
from my_libraries import *

capstone_folder, images_folder, images_100x100_folder = folders()
if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''
cur = conn.cursor()

def doquery(curs, query, title):
    #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
    try:
        curs.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n************************* error in {}:\n{}'.format(title, ex))
        cur.connection.close()
        sys.exit()

#get paths of images to process
query = '''
SELECT summit_id, image_id, filename FROM images ORDER BY image_id --LIMIT 5;
'''
doquery(cur, query, "select filenames")
images_paths = cur.fetchall()

df = pd.DataFrame(columns=['summit_id','image_id','shape'])

errors = []
# row = 0
# image = (16669, 1, 'CA_0_16669_1.jpg')
for row, image in enumerate(images_paths):
    # print(image)
    if row % 100 == 0:
        print("row# {}".format(row))
    summit_id, image_id, filename = image
    try:
        img = io.imread(images_folder +filename)
    except Exception as ex:
        print("error reading summit_id {}, image {}".format(summit_id, image_id))
        errors.append((summit_id, image_id, "read"))
    if img.shape[0] < 20 or img.shape[1] < 20:
        print("summit_id {}, image_id {}, img.shape ={})".format(summit_id, image_id, img.shape))
        errors.append((summit_id, image_id, "size"))

    newrow = {'summit_id': summit_id, 'image_id': image_id, 'shape': img.shape}
    df = df.append(newrow, ignore_index=True)

    resized_img = resize(img, (100,100), clip=False)
    io.imsave(images_100x100_folder + filename, resized_img)


if len(errors) > 0:
    with open(capstone_folder + "still_errors.pkl", 'wb') as f:
        pickle.dump(errors, f)
else:
    print("all images downloaded OK")

# with open(capstone_folder + "df_images.pkl", 'wb') as f:
#     pickle.dump(df, f)
