'''
This was initially used before ANFIS. The idea was to make a lookup table using a SQL database for fast lookup results. 

This script converts a CSV file, as a table with all kinematics of the workspace mapped, to a SQL file
'''

import csv, sqlite3

# create the SQL file
sqlite_file = 'kinematics_db.sqlite'
con = sqlite3.connect(sqlite_file)
cur = con.cursor()

cur.execute("CREATE TABLE t (x REAL, y REAL, z REAL, theta1 REAL, theta2 REAL, theta3 REAL);") # use your column names here

with open('forward_kinematics_2.csv','rb') as fin: # `with` statement available in 2.5+
    # csv.DictReader uses first line in file for column headings by default
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['x'], i['y'], i['z'], i['theta1'], i['theta2'], i['theta3']) for i in dr]

cur.executemany("INSERT INTO t (x, y, z, theta1, theta2, theta3) VALUES (?, ?, ?, ?, ?, ?);", to_db)
con.commit()
con.close()