import os
import psycopg2
import sys

home = os.getenv("HOME")
capstone_folder = home + "/dsi/Capstone/"
images_folder = home + "/dsi/Capstone-images/"

if home == '/Users/edwardwhite': #MacBook
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed': #Linux
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='ed', password=p)
    p = None
cur = conn.cursor()

def folders():
    return capstone_folder, images_folder

def psql_conn():
    return cur, conn

def doquery(cur, query, title):
    #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
    try:
        cur.execute(query)
        cur.connection.commit()
    except Exception as ex:
        print('\n************************* error in {}:\n{}'.format(title, ex))
        cur.connection.close()
        sys.exit()

def save_to_pickle(data, filename):
    '''
    INPUT:
    data: any format that can be pickled
    text: filename to save picked file to

    OUTPUT: none

    Save data in pickle format to filename
    '''
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filename):
    '''
    INPUT: text
    OUTPUT: data retrieved from pickle file
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)
