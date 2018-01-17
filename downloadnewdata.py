from urllib import request
import pandas as pd
import os
import psycopg2
import sys

def get_save_image(summit_id, image_id, df_row, type_str):
    if type_str == None:
        type_str = "None"

    query = '''SELECT EXISTS (SELECT image_id FROM images WHERE image_id={})'''.format(image_id)
    doquery(cur, query, 'image_id {} exists'.format(image_id))
    in_images = cur.fetchone()[0]

    fn = images_folder + type_str.title() + "/" + str(summit_id) + "_" + str(image_id) + ".jpg"
    if os.path.isfile(fn):
        image_filesaved = True
    else:
        image_filesaved = False

    if image_filesaved == True:
        if in_images == True:
            print("image# {} is download and in db")
            return

        else: #file downloaded but not in db--should never happen
            print("******************************* image# {} file downloaded but image is not in images table")
            sys.exit()

    #image not downloaded, so download it ::::::::::::::::::::::::::
    #delete any entry in db that may already exist for this image_id
    query = '''DELETE FROM images WHERE image_id={};'''.format(image_id)
    doquery(cur, query, "delete image# {} data from images".format(image_id))

    url = df_row['link']
    with open(fn, 'wb') as f:
        f.write(request.urlopen(url).read())

    query = '''INSERT INTO images(summit_id, image_id, url, filepath) VALUES ({}, {}, '{}', '{}')'''.format(summit_id, image_id, url, fn)
    doquery(cur, query, "insert into images")
    print("Downloaded image# {}".format(image_id))


def gettype(name):
    #A few summits have multiple types in the name, e.g. Mount Ellen Peak or Big Peaked Mountain--classify these as None
    numcounted = 0

    if 'Mount' in name and 'Mountain' not in name:
        type = 0
        type_str = "mount"
        numcounted += 1

    if 'Mountain' in name:
        type = 1
        type_str = "mountain"
        numcounted +=1

    if "Peak" in name:
        type = 2
        type_str = "peak"
        numcounted += 1

    if numcounted != 1:
        type = 4
        type_str = None

    return type, type_str
    

def doquery(curs, query, title):
    #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
    try:
        curs.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n********************************************* error in {}\n{}'.format(title, ex))
        cur.connection.close()
        sys.exit()


home = os.getenv("HOME")
capstone_folder = home + "/dsi/Capstone/"
images_folder = capstone_folder + "images/"
excel_file = capstone_folder +"/listsofjohn/all_loj_images.xlsx"
df = pd.read_excel(excel_file)
df.sort_values('image_id', inplace=True) #so row order is consistent every time this is run

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''

cur = conn.cursor()

# row = 22 #for testing purposes
for row in range(0, df.shape[0]):
    print("row# {} ".format(row), end='')
# for row in range(20): #test
    df_row = df.iloc[row]
    summit_id = df_row['summit_id'])
    image_id = df_row['image_id']
    type, type_str = gettype(str(df_row['Name'])) #some names are just numbers, which are integers in the Excel file, so use str() to convert

     #if counties has an apostrophe in it, make it two apostrophes for SQL to work
     #state is simply 2 letter abbreviation, so no apostrophes
    counties = df_row['Counties'].replace("'", "''")
    quad = df_row['Quad'].replace("'", "''")

    query = '''SELECT EXISTS (SELECT summit_id FROM summits WHERE summit_id={});'''.format(summit_id)
    doquery(cur, query, "check if summit exists")
    try:
        indb = cur.fetchone()
    except Exception as ex:
        print("\n********************************************* db error: {}".format(ex))
        sys.exit()

    if indb[0] == True: #this summit is already in db
        print("summit_id {} already in db. ".format(summit_id), end='')

        #confirm state matches
        query='''SELECT state FROM summits WHERE summit_id={} ;       '''.format(summit_id)
        doquery(cur, query, 'get state from db for id={}'.format(summit_id))
        db_state = cur.fetchone()[0]
        if db_state != df_row['State']:
            print("summit_id={}: excel state={} != db state={} **************************************** ".format(summit_id, df_row['State'], db_state))
            query = '''UPDATE summits SET state='{}', counties='{}', quad='{}' WHERE summit_id={};'''.format(df_row['State'], counties, quad, summit_id)
            doquery(cur, query, 'update state for summit_id: {}'.format(summit_id))
        else:
            print("State matches. ", end='')

            query = '''UPDATE summits SET counties='{}', quad='{}' WHERE summit_id={};'''.format(counties, quad, summit_id)
            doquery(cur, query, "update counties for {}".format(summit_id))

        get_save_image(summit_id, image_id, df_row, type_str)

    else: #data is not in db, so add it
        print("adding data for summit_id {}. ".format(summit_id), end='')
        #use '' in replace of ' for updating db
        name = str(df_row['Name']).replace("'", "''") #needed for SQL INSERT

        longitude = df_row['Longitude']
        if longitude < 0: longitude =- longitude #all longitude should be negative for U.S., but some in the Excel file provided are incorrectly positive

        query = '''INSERT INTO summits (summit_id, name, elevation, isolation, prominence, latitude, longitude, type, counties, state, type_str, quad)  VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');'''.format(df_row['summit_id'], name, df_row['Elevation'], df_row['Isolation'], df_row['Prominence'], df_row['Latitude'], longitude, type, counties, df_row['State'], type_str, quad)
        doquery(cur, query, "insert new summit# {} into summits".format(summit_id))

        get_save_image(summit_id, image_id, df_row, type_str)

conn.close()

print("\n!!!!!!!!!!!!!!!!!!!!! FINISHED! !!!!!!!!!!!!!!!!!!!!!!!!!")
