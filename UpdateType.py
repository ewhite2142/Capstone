import os
import psycopg2
import sys


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

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    conn_u = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''

cur = conn.cursor()
cur_u = conn_u.cursor()

#can only get max 2500 locations from google each day; 7161 total locations
query = '''SELECT name, summit_id FROM summits ORDER BY summit_id;''' #no ";" at end because query used in countquery below
doquery(cur, query, 'collect all rows')
numrows = cur.rowcount
# print("numrows={}".format(numrows))
for i, record in enumerate(cur):
    # print("i={}, record={}".format(i, record))
    try:
        name, summit_id = record
    except Exception as ex:
        print('\n***** error trying to retrieve name, etc. summit_id={}\n{}'.format(summit_id, ex))
        conn.close()
        sys.exit()

    type, type_str = gettype(name)


    query = '''UPDATE summits SET type=%s, type_str=%s WHERE summit_id=%s;'''
    cur_u.execute(query, (type, type_str, summit_id))
    conn_u.commit()

conn.close()

gettype("Holy Cross, Mount of the")
name = "Holy Cross, Mount of the"
summit_id = 64
