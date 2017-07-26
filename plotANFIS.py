from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy as np
import time
import cPickle as pickle
from fuzzyErrorTest import fuzzy_error_test
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



with open('fuzzy_log_test_table_90_270_5_epoch_40.pkl', 'rb') as f:
    anf = pickle.load(f)

# total_error, average_error = fuzzy_error_test(anf, "test_table_176_184.csv")
#
# print "The total error is: ", total_error
# print "The average error is: ", average_error

# var = numpy.array([[3, 4], [3, 4]])
var = np.array([[0,0,-207]])

print "input", var

# print "the length is", len(var[:,0])
#
# var = numpy.asanyarray(var)
# print "the shape", var.shape
# print "the shape index", var.shape[0]
# print "the shape index", var.shape[1]

# print the predicted value based on the trained set
print "The answer is", anfis.predict(anf, var)

anf.plotErrors()
anf.plotResults()