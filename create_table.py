import psycopg2
try:
    conn = psycopg2.connect(dbname='postgres', host='localhost')
    cur = conn.cursor()
    conn.autocommit = True
    cur.execute('DROP DATABASE IF EXISTS summitsdb;')
    cur.execute('CREATE DATABASE summitsdb;')
    cur.close() # This is optional
    conn.close() # Closing the connection also closes all cursors
except Exception as ex:
    print('error creating db:\n{}'.format(ex))
    conn.close()


try:
    conn = psycopg2.connect(dbname='summitsdb', host='localhost')
    cur = conn.cursor()

    query = '''
    CREATE TABLE summits (
    summit_id      int,
    image_id    int     PRIMARY KEY,
    name        varchar(32),
    elevation   int,
    isolation   numeric(6,2),
    prominence  int,
    latitude    numeric(8,5),
    longtitude  numeric(8,5),
    type        varchar(8),
    url         varchar(40),
    filepath    varchar(41)
    );
    '''
    cur.execute(query)
    conn.commit()
except Exception as ex:
    print('error creating table:\n{}'.format(ex))
    conn.close()

try:
    query = '''
            COPY summits
            FROM '/users/edwardwhite/dsi/Capstone/summits.csv'
            DELIMITER ','
            CSV HEADER;
            '''
    cur.execute(query)
    conn.commit()
except Exception as ex:
    print('error copying table:\n{}'.format(ex))
    conn.close()

cur.close()
conn.close()
