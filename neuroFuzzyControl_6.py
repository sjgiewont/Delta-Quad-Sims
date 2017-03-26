from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy as np
import time
import cPickle as pickle
from fuzzyErrorTest import fuzzy_error_test
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
ts = np.loadtxt("small_test_table_170_190_2.csv", delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

# using coord input, theta output
# X = ts[:,0:3]
# Y = ts[:,3:7]

X = ts[:,0:3]
Y = ts[:,3:6]

print X
print Y

x = ts[:, 0]
y = ts[:, 1]
z = ts[:, 2]

x_step = (np.max(x) - np.min(x)) / 4
x_start = np.min(x) + (x_step / 2)
x_sigma = x_step / 2

y_step = (np.max(y) - np.min(y)) / 4
y_start = np.min(y) + (y_step / 2)
y_sigma = y_step / 2

z_step = (np.max(z) - np.min(z)) / 4
z_start = np.min(z) + (z_step / 2)
z_sigma = z_step / 2



print x_step, x_start, x_sigma, np.max(x), np.min(x)
print y_step, y_start, y_sigma, np.max(y), np.min(x)
print z_step, z_start, z_sigma, np.max(z), np.min(z)

# X = numpy.matrix('1 1 1; 2 2 2; 3 3 3')
# Y = numpy.array([6, 3, 1])


x_mu_1 = x_start
x_mu_2 = x_start + x_step
x_mu_3 = x_start + 2 * x_step
x_mu_4 = x_start + 3 * x_step
x_width_1 = 10*x_sigma
x_slope_1 = 1

y_mu_1 = y_start
y_mu_2 = y_start + y_step
y_mu_3 = y_start + 2 * y_step
y_mu_4 = y_start + 3 * y_step
y_width_1 = 10*y_sigma
y_slope_1 = 1

z_mu_1 = z_start
z_mu_2 = z_start + z_step
z_mu_3 = z_start + 2 * z_step
z_mu_4 = z_start + 3 * z_step
z_width_1 = 10*z_sigma
z_slope_1 = 1

mf = [[['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_1}],
       ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_2}],
       ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_3}],
       ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_4}]],
      [['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_1}],
       ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_2}],
       ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_3}],
       ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_4}]],
      [['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_1}],
       ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_2}],
       ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_3}],
       ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_4}]]]

x = np.linspace(np.min(x)+np.min(x)/2, np.max(x)+np.max(x)/2, 100)
z_linspace = np.linspace(np.min(z)-z_sigma, np.max(z)+z_sigma, 100)

print "Starting membership function"
mfc = membership.membershipfunction.MemFuncs(mf)

print "Finished membership function, starting ANFIS"
anf = anfis.ANFIS(X, Y, mfc)

# anf.plotMF(x, 0)
# anf.plotMF(x, 1)
# anf.plotMF(z_linspace, 2)

t = time.time()

print "Finished ANFIS, starting training Hybrid"
anf.trainHybridJangOffLine(epochs=5)

with open('small_fuzzy_test_bell.pkl', 'wb') as f:
    pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

print time.time() - t

total_error, average_error = fuzzy_error_test(anf, "test_table_176_184.csv")

print "The total error is: ", total_error
print "The average error is: ", average_error

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

# anf.plotErrors()
# anf.plotResults()
