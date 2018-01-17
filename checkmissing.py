from urllib import request
import pandas as pd
import os
import psycopg2
import sys

home = os.getenv("HOME")
capstone_folder = home + "/dsi/Capstone/"
images_folder = capstone_folder + "images/"
excel_file = capstone_folder +"/listsofjohn/all_loj_images.xlsx"
df = pd.read_excel(excel_file)

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
        conn.commit()
    except Exception as ex:
        print('error in {}\n{}'.format(title, ex))
        conn.close()
        sys.exit()

summitnotin = []
imagenotin = []
for row in range(0, df.shape[0]):
    df_row = df.iloc[row]
    summit_id = df_row['summit_id']
    image_id = df_row['image_id']

    query='''SELECT EXISTS (SELECT summit_id FROM images WHERE summit_id={});'''.format(summit_id)
    doquery(cur, query, "find summit_id")
    if cur.fetchone()[0] == False:
        print("summit_id {}, {} is missing".format(summit_id, df_row['Name']))
        summitnotin.append(summit_id)

    # query='''SELECT EXISTS (SELECT image_id FROM images WHERE image_id={});'''.format(image_id)
    # doquery(cur, query, "find image_id")
    # if cur.fetchone()[0] == False:
    #     print("image_id {}, {} is missing".format(image_id, df_row['Name']))
    #     imagenotin.append(image_id)
