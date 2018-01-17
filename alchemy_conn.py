from sqlalchemy import create_engine
import getpass
import os
import psycopg2

home = os.getenv("HOME")
with open(home +  '/.google/psql', 'r') as f:
    p = f.readline().strip()


if home == '/Users/edwardwhite':
    username = 'edwardwhite'

elif home == '/home/ed':
    username = 'postgres'

temp = 'postgresql+psycopg2://' + username + ':' + p + '@localhost/summitsdb'
# print(temp)
def alchemy_engine():
        engine = create_engine(temp)
        return engine.connect()
