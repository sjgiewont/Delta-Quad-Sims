import numpy as np
import time
import csv

ts = np.loadtxt("table_150_211_one.csv", delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

Z = ts[:, 2]

min_Z = np.min(Z)
max_Z = np.max(Z)

start = -170
step = 5

end_range = start - step


while min_Z - step < end_range:
    Z_index = np.array(np.where((Z <= start) & (Z > end_range)))

    filename = "test_140_220_qrtr_%d_%d.csv" % (abs(start), abs(end_range))
    print filename

    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(Z_index.size):
            row = ts[Z_index[0][i], 0:6]
            spamwriter.writerow(row)

    start -= step
    end_range -= step