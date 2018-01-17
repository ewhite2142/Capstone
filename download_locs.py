'''read counties and state from google geocoding for each summit_id in summits table'''
import psycopg2
import datetime as dt
import googlemaps
import os


def doquery(curs, query, title):
    #best to close conn if error (error leaves conn open), cuz multiple conn open causes problems
    try:
        curs.execute(query)
        conn.commit()
    except Exception as ex:
        print('error in {}\n{}'.format(title, ex))
        conn.close()

#open google api for locations
apikey = os.environ["GOOGLE_APIKEY"]
gmaps = googlemaps.Client(key=apikey)

#different conn if on Mac or Ubuntu
home = os.getenv("HOME")
if home == '/Users/edwardwhite':
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')
    conn_u = psycopg2.connect(dbname='summitsdb', host='localhost')

elif home == '/home/ed':
    with open(home +  '/.google/psql', 'r') as f:
        p = f.readline().strip()
    conn = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    conn_u = psycopg2.connect(dbname='summitsdb', host='localhost', user='postgres', password=p)
    p=''

# one DB conn & cursor needed to loop through each summit_id with fetchone()
#second different conn & cursor needed for updating DB
cur = conn.cursor()
cur_u = conn.cursor()

#can only get max 2500 locations from google each day; 7161 total locations
query = '''SELECT summit_id, latitude, longitude FROM summits WHERE state is NULL OR counties IS NULL ORDER BY summit_id LIMIT 2500''' #no ";" at end because query used in countquery below
countquery = '''SELECT COUNT(*) FROM (''' + query + ''') x;'''
# print("countquery={}".format(countquery))
doquery(cur, countquery, 'count summits')
numrows = cur.fetchone()[0] #fetchone returns tuple
print("numrows={}".format(numrows))

query += ';'
# print("query={}".format(query))
doquery(cur, query, 'summits')

#loop through numrows of summit_id to get counties and state for each
for i in range(numrows):
    print("i={}".format(i))
    summit_id, latitude, longitude = cur.fetchone()
    print("summit_id, latitude, longitude={}, {}, {}".format(summit_id, latitude, longitude))

    loc = gmaps.reverse_geocode((latitude, longitude))
    #loc is a complicated and inconsistent list of dicts
    #search 'types' in loc for state and counties, as is inconsistent where they are in loc for each lookup
    for i in range(len(loc)):
        if loc[i]['types'][0] == 'administrative_area_level_1':
            state = loc[i]['address_components'][0]['short_name'] # short_name gives 2 letter state name, e.g "CO"
        elif loc[i]['types'][0] == 'administrative_area_level_2':
            counties = loc[i]['address_components'][0]['long_name']
    print("counties: {}, state: {}".format(counties, state))
    print()

     #if counties or state has an apostrophe in it, make it two apostrophes for SQL to work
    counties = counties.replace("'", "''")
    state = state.replace("'", "''")

    #update counties and state in db for this summit_id
    query = '''
    UPDATE summits
    SET counties = '{}', state = '{}'
    WHERE summit_id = {};
    '''.format(counties, state, summit_id)

    # print("query={}".format(query))
    doquery(cur_u, query, 'update summit_id# {}'.format(summit_id))
