from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy as np
import time
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
ts = np.loadtxt("table_175_185.csv", delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

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

y_mu_1 = y_start
y_mu_2 = y_start + y_step
y_mu_3 = y_start + 2 * y_step
y_mu_4 = y_start + 3 * y_step

z_mu_1 = z_start
z_mu_2 = z_start + z_step
z_mu_3 = z_start + 2 * z_step
z_mu_4 = z_start + 3 * z_step

mf = [[['gaussmf', {'mean': x_mu_1, 'sigma': x_sigma}], ['gaussmf', {'mean': x_mu_2, 'sigma': x_sigma}],
       ['gaussmf', {'mean': x_mu_3, 'sigma': x_sigma}],
       ['gaussmf', {'mean': x_mu_4, 'sigma': x_sigma}]],
      [['gaussmf', {'mean': y_mu_1, 'sigma': y_sigma}], ['gaussmf', {'mean': y_mu_2, 'sigma': y_sigma}],
       ['gaussmf', {'mean': y_mu_3, 'sigma': y_sigma}],
       ['gaussmf', {'mean': y_mu_4, 'sigma': y_sigma}]],
      [['gaussmf', {'mean': z_mu_1, 'sigma': z_sigma}], ['gaussmf', {'mean': z_mu_2, 'sigma': z_sigma}],
       ['gaussmf', {'mean': z_mu_3, 'sigma': z_sigma}],
       ['gaussmf', {'mean': z_mu_4, 'sigma': z_sigma}]]]

x = np.linspace(np.min(x)+np.min(x)/2, np.max(x)+np.max(x)/2, 100)

print "Starting membership function"
mfc = membership.membershipfunction.MemFuncs(mf)

print "Finished membership function, starting ANFIS"
anf = anfis.ANFIS(X, Y, mfc)

anf.plotMF(x, 0)

t = time.time()

print "Finished ANFIS, starting training Hybrid"
anf.trainHybridJangOffLine(epochs=5)

with open('fuzzy_test_gauss.pkl', 'wb') as f:
    pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

print time.time() - t

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

