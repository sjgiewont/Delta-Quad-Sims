'''
Used to search an SQL database and time how long it takes
'''

import sqlite3
import time

sqlite_file = 'kinematics_db.sqlite'
con = sqlite3.connect(sqlite_file)
cur = con.cursor()

x = 2.5
y = 10.9
z = -118

# cur.execute('SELECT * FROM "t" WHERE x=0 AND y=0 ORDER BY ABS( z - (-94) ) ')
# cur.execute('SELECT * FROM "t" WHERE abs(x - (0)) = (SELECT min(abs(x - (0))) FROM "t") AND abs(y - (.34)) = (SELECT min(abs(y - (.34))) FROM "t") AND abs(z - (-94)) = (SELECT min(abs(z - (-94))) FROM "t")')

tu = (x, y, z)

# command_string = 'SELECT * FROM "t" WHERE abs(x - ({0})) = (SELECT min(abs(x - ({0}))) FROM "t") AND abs(y - ({1})) = (SELECT min(abs(y - ({1}))) FROM "t") AND abs(z - ({2})) = (SELECT min(abs(z - ({2}))) FROM "t")' .format(*tu)
command_string = 'SELECT * FROM "t" WHERE x = {0} AND y = {1} AND z = {2}' .format(*tu)

t = time.time()

cur.execute(command_string)

# cur.execute('SELECT * FROM "t" WHERE abs(x - (0)) = (SELECT min(abs(x - (0))) FROM "t") AND abs(y - (2.88)) = (SELECT min(abs(y - (2.88))) FROM "t") AND abs(z - (-100)) = (SELECT min(abs(z - (-100))) FROM "t")')
# cur.execute('SELECT * FROM "t" WHERE abs(y - (2.88)) = (SELECT min(abs(y - (2.88))) FROM "t")')
# cur.execute('SELECT * FROM "t" WHERE abs(z - (-100)) = (SELECT min(abs(z - (-100))) FROM "t")')

ans = cur.fetchone()
print ans
print time.time() - t
