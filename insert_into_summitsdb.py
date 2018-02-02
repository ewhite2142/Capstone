'''Inserts data from listsofjohn Excel file into the summits and images tables in summitsdb'''

import pandas as pd
import os
import psycopg2
import sys


def doquery(cur, query, errmsg):
    '''
    INPUT
    cur: cursor to the PSQL database
    query: SQL query string
    errmsg: string to display if an error occurs during query execution

    When an error occurs during query execution, it usually leaves the DB connection open, and having too many open connections can cause a problem. This function uses try-except and closes the connection if an error occurs.
    '''
    try:
        cur.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n********************************************* error in {}\n{}'.format(errmsg, ex))
        cur.connection.close()
        sys.exit()


home = os.getenv("HOME")
capstone_folder = home + "/dsi/Capstone/"
images_folder = capstone_folder + "images/"
excel_file = capstone_folder +"/listsofjohn/all_loj_images.xlsx"

print("Reading excel data into DataFrame...")
df = pd.read_excel(excel_file)
print("Finished reading Excel data.")
df.sort_values('image_id', inplace=True) #so row order is consistent every time this is run

#get DB cursor and connection
if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''

cur = conn.cursor()

print("Deleting exising db data...", end='')
query = '''
DELETE FROM images;
DELETE FROM summits;
'''
doquery(cur, query, "delete all")
print("finished.")
summits_list = []

#loop through each row in Excel file and insert into tables
for row in range(0, df.shape[0]):
    # if row > 5: break #TESTING
    if row % 100 == 0:
        print("row# {} ".format(row))

    df_row = df.iloc[row]
    summit_id = df_row['summit_id']
    image_id = df_row['image_id']

    if summit_id not in summits_list: #each summit could be listed more than once because could have multiple images
        summits_list.append(summit_id)
        # print("adding summit_id {}".format(summit_id))

        #add to summits table
        type, type_str = gettype(str(df_row['Name'])) #some names are just numbers, which are integers in the Excel file, so use str() to convert

         #if counties has an apostrophe in it, make it two apostrophes for SQL to work
         #state is simply 2 letter abbreviation, so no apostrophes
        counties = df_row['Counties'].replace("'", "''")
        quad = df_row['Quad'].replace("'", "''")

        #use '' in replace of ' for updating db
        name = str(df_row['Name']).replace("'", "''") #needed for SQL INSERT

        longitude = df_row['Longitude']
        if longitude > 0: longitude =- longitude #all longitude should be negative for U.S., but some in the Excel file provided are incorrectly positive

        query = '''
        INSERT INTO summits (summit_id, name, elevation, isolation, prominence, latitude, longitude, type, type_str, counties, states, quad)
        VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');
        '''.format(df_row['summit_id'], name, df_row['Elevation'], df_row['Isolation'], df_row['Prominence'], df_row['Latitude'], longitude, type, type_str, counties, df_row['State'], quad)
        doquery(cur, query, "insert new summit# {} into summits".format(summit_id))

    #add to images table
    # print("adding image_id {}".format(image_id))
    if type_str == None:
        type_str = "None"

    url = df_row['link']
    query = '''INSERT INTO images(summit_id, image_id, url, filename) VALUES ({}, {}, '{}', '{}')'''.format(summit_id, image_id, url, str(image_id) + ".jpg")
    doquery(cur, query, "insert into images")

# use query below to definitely set type, type_str in summits table
query = '''
UPDATE summits SET type=5, type_str='None';

UPDATE summits SET type=0, type_str='mount'
WHERE name LIKE '%Mount%' AND name NOT LIKE '%Mountain%' AND name NOT LIKE '%Mountaineer'
    AND  ( name NOT LIKE '%Peak%'
        OR name LIKE '%-North Peak%'
        OR name LIKE '%-South Peak%'
        OR name LIKE '%-East Peak%'
        OR name LIKE '%-West Peak%'
        OR name LIKE '%-Middle Peak%'
        OR name LIKE '%-Southwest Peak%' );

UPDATE summits SET type=1, type_str='mountain'
WHERE name LIKE '%Mountain%' AND name NOT LIKE '%Mountaineer'
AND  ( name NOT LIKE '%Peak%'
    OR name LIKE '%-North Peak%'
    OR name LIKE '%-South Peak%'
    OR name LIKE '%-East Peak%'
    OR name LIKE '%-West Peak%'
    OR name LIKE '%-Middle Peak%'
    OR name LIKE '%-Southwest Peak%' );

UPDATE summits SET type=2, type_str='peak'
WHERE name LIKE '%Peak%' AND name NOT LIKE '%Mount%'
    AND name NOT LIKE '%-North Peak%'
    AND name NOT LIKE '%-South Peak%'
    AND name NOT LIKE '%-East Peak%'
    AND name NOT LIKE '%-West Peak%'
    AND name NOT LIKE '%-Middle Peak%'
    AND name NOT LIKE '%-Southwest Peak%';

UPDATE summits SET type=4, type_str='ambiguous' WHERE summit_id IN
(
SELECT summit_id FROM summits
WHERE type = 5 AND (
    name LIKE '%Mount%'  OR
    name LIKE '%Peak%'
                    )
    AND name NOT LIKE '%-North Peak%'
    AND name NOT LIKE '%-South Peak%'
    AND name NOT LIKE '%-East Peak%'
    AND name NOT LIKE '%-West Peak%'
    AND name NOT LIKE '%-Middle Peak%'
    AND name NOT LIKE '%-Southwest Peak%'
);

UPDATE summits SET type=2, type_str='peak'
WHERE name LIKE '%Mountaineer Peak%';
'''
doquery(cur, query, "set titles")

conn.close()

print("\n!!!!!!!!!!!!!!!!!!!!! FINISHED! !!!!!!!!!!!!!!!!!!!!!!!!!")
