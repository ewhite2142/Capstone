from urllib import request
from skimage import io
import os
import psycopg2
import sys
from my_libraries import folders
import _pickle as pickle

capstone_folder, images_folder, images_100x100_folder = folders()
home = os.getenv("HOME")

def doquery(curs, query, title):
    #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
    try:
        curs.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n************************* error in {}:\n{}'.format(title, ex))
        cur.connection.close()
        sys.exit()

with open(capstone_folder + "errors_list.pkl", 'rb') as f:
    errors = pickle.load(f)

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''
cur = conn.cursor()

newerrors = []
max_tries = 1
error = (2152, 1000, 'read') #for testing purposes
for error in errors:
    summit_id = error[0]
    image_id = error[1]

    query = '''SELECT filename, url FROM images WHERE image_id={};'''.format(image_id)
    doquery(cur, query, "{}, {} get filename from images".format(summit_id, image_id))
    try:
        image_filename, url = cur.fetchone()
    except Exception as ex:
        print("\n******** db error {}, {}: {}".format(summit_id, image_id, ex))
        sys.exit()

    image_shortname = image_filename
    image_filename = images_folder + image_filename
    os.remove(image_filename) #delete bad file
    numtries = 0
    notread = True
    while notread == True:
        numtries += 1
        if numtries > max_tries:
            print("{}, {}, {}, {}".format(summit_id, image_id, image_shortname, url))
            newerrors.append((summit_id, image_id, image_filename))
            break

        with open(image_filename, 'wb') as f:
            newfile = request.urlopen(url).read()
            f.write(newfile)

        #confirm image downloaded correctly by reading it
        try:
            img = io.imread(image_filename)
        except Exception as ex:
            continue

        if img.shape[0] < 20 or img.shape[1] < 20:
            continue

        notread = False

    if notread == False:
        print("Downloaded image# {}".format(image_id))

conn.close()
with open(capstone_folder + "stubborn_errors.pkl", 'wb') as f:
        pickle.dump(newerrors, f)

print("\n!!!!!!!!!!!!!!!!!!!!! FINISHED! !!!!!!!!!!!!!!!!!!!!!!!!!")
