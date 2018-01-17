import os
import psycopg2
import sys


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

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    # conn_u = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''

cur = conn.cursor()
# cur_u = conn_u.cursor()

for file in os.listdir(images_folder):
    underscore = file.find("_")
    period = file.find(".")
    summit_id = file[:underscore]
    image_id = file[underscore+1:period]

    query = '''SELECT DISTINCT s.state, s.type FROM images i INNER JOIN summits s ON s.summit_id = i.summit_id WHERE i.image_id={};'''.format(image_id)
    doquery(cur, query, "select state, type from images")

    state, type = cur.fetchone()

    newfilename = state + "_" + type + "_" + file
    # print (images_folder + file + " --> " + images_folder + newfilename)
    os.rename(images_folder + file, images_folder + newfilename)

    query = '''UPDATE images SET filename = '{}' WHERE image_id={};'''.format(newfilename, image_id)
    doquery(cur, query, "update filename")

conn.close()
