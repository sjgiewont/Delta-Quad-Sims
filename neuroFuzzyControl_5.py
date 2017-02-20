from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import time
import cPickle as pickle

# ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
ts = numpy.loadtxt("table_175_185.csv", delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

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

x_step = (numpy.max(x) - numpy.min(x)) / 4
x_start = numpy.min(x) + (x_step / 2)
x_sigma = x_step / 2

y_step = (numpy.max(y) - numpy.min(y)) / 4
y_start = numpy.min(y) + (y_step / 2)
y_sigma = y_step / 2

z_step = (numpy.max(z) - numpy.min(z)) / 4
z_start = numpy.min(z) + (z_step / 2)
z_sigma = z_step / 2

# X = numpy.matrix('1 1 1; 2 2 2; 3 3 3')
# Y = numpy.array([6, 3, 1])

# mf = [[['gaussmf',{'mean':-11.,'sigma':5.}],['gaussmf',{'mean':-8.,'sigma':5.}],['gaussmf',{'mean':-14.,'sigma':20.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
#             [['gaussmf',{'mean':-10.,'sigma':20.}],['gaussmf',{'mean':-20.,'sigma':11.}],['gaussmf',{'mean':-9.,'sigma':30.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
#             [['gaussmf', {'mean': -10., 'sigma': 20.}], ['gaussmf', {'mean': -20., 'sigma': 11.}], ['gaussmf',{'mean': -9.,'sigma': 30.}], ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]]

# mf = [[['gaussmf',{'mean':-5.,'sigma':5.}],['gaussmf',{'mean':0.,'sigma':5.}],['gaussmf',{'mean':5.,'sigma':5.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
#             [['gaussmf',{'mean':-5.,'sigma':5.}],['gaussmf',{'mean':0.,'sigma':5.}],['gaussmf',{'mean':5.,'sigma':5.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
#             [['gaussmf', {'mean': -100., 'sigma': 20.}], ['gaussmf', {'mean': -120., 'sigma': 10.}], ['gaussmf',{'mean': -130.,'sigma': 30.}], ['gaussmf', {'mean': -94, 'sigma': 5}]]]

# mf = [[['gaussmf',{'mean':-5.,'sigma':2.}],['gaussmf',{'mean':0.,'sigma':2.}],['gaussmf',{'mean':5.,'sigma':2}],['gaussmf',{'mean':-7.,'sigma':7.}]],
#             [['gaussmf',{'mean':-5.,'sigma':2.}],['gaussmf',{'mean':0.,'sigma':2.}],['gaussmf',{'mean':5.,'sigma':2.}],['gaussmf',{'mean':-10.5,'sigma':5.}]],
#             [['gaussmf', {'mean': -100., 'sigma': 10.}], ['gaussmf', {'mean': -120., 'sigma': 10.}], ['gaussmf',{'mean': -130.,'sigma': 10.}], ['gaussmf', {'mean': -94, 'sigma': 5}]]]

# mf = [[['gaussmf',{'mean':-20.,'sigma':10.}],['gaussmf',{'mean':0.,'sigma':10.}],['gaussmf',{'mean':20.,'sigma':10.}]],
#             [['gaussmf',{'mean':-20.,'sigma':10.}],['gaussmf',{'mean':0.,'sigma':10.}],['gaussmf',{'mean':20.,'sigma':10.}]],
#             [['gaussmf', {'mean': -100., 'sigma': 5.}], ['gaussmf', {'mean': -110., 'sigma': 5.}], ['gaussmf',{'mean': -120.,'sigma': 5.}]]]

# mf = [[['gaussmf',{'mean':-40.,'sigma':10.}],['gaussmf',{'mean':-20.,'sigma':10.}],['gaussmf',{'mean':10.,'sigma':10.}],['gaussmf',{'mean':30.,'sigma':10.}]],
#             [['gaussmf',{'mean':-41.,'sigma':10.}],['gaussmf',{'mean':-21.,'sigma':10.}],['gaussmf',{'mean':12.,'sigma':10.}],['gaussmf',{'mean':31,'sigma':10.}]]]

# mf = [[['gaussmf', {'mean': x_start, 'sigma': x_sigma}], ['gaussmf', {'mean': x_start + x_step, 'sigma': x_sigma}],
#        ['gaussmf', {'mean': x_start + 2 * x_step, 'sigma': x_sigma}],
#        ['gaussmf', {'mean': x_start + 3 * x_step, 'sigma': x_sigma}]],
#       [['gaussmf', {'mean': y_start, 'sigma': y_sigma}], ['gaussmf', {'mean': y_start + y_step, 'sigma': y_sigma}],
#        ['gaussmf', {'mean': y_start + 2 * y_step, 'sigma': y_sigma}],
#        ['gaussmf', {'mean': y_start + 3 * y_step, 'sigma': y_sigma}]],
#       [['gaussmf', {'mean': z_start, 'sigma': z_sigma}], ['gaussmf', {'mean': z_start + z_step, 'sigma': z_sigma}],
#        ['gaussmf', {'mean': z_start + 2 * z_step, 'sigma': z_sigma}],
#        ['gaussmf', {'mean': z_start + 3 * z_step, 'sigma': z_sigma}]]]

# mf = [[['sigmf', {'b': x_start, 'c': x_sigma}], ['sigmf', {'b': x_start + x_step, 'c': x_sigma}],
#        ['sigmf', {'b': x_start + 2 * x_step, 'c': x_sigma}],
#        ['sigmf', {'b': x_start + 3 * x_step, 'c': x_sigma}]],
#       [['sigmf', {'b': y_start, 'c': y_sigma}], ['sigmf', {'b': y_start + y_step, 'c': y_sigma}],
#        ['sigmf', {'b': y_start + 2 * y_step, 'c': y_sigma}],
#        ['sigmf', {'b': y_start + 3 * y_step, 'c': y_sigma}]],
#       [['sigmf', {'b': z_start, 'c': z_sigma}], ['sigmf', {'b': z_start + z_step, 'c': z_sigma}],
#        ['sigmf', {'b': z_start + 2 * z_step, 'c': z_sigma}],
#        ['sigmf', {'b': z_start + 3 * z_step, 'c': z_sigma}]]]

# mf = [[['sigmf', {'b': 0, 'c': 0.25}], ['sigmf', {'b': 0, 'c': -0.25}], ['sigmf', {'b': 0, 'c': 0.5}], ['sigmf', {'b': 0, 'c': -0.5}]],
#       [['sigmf', {'b': 0, 'c': 0.25}], ['sigmf', {'b': 0, 'c': -0.25}], ['sigmf', {'b': 0, 'c': 0.5}], ['sigmf', {'b': 0, 'c': -0.5}]],
#       [['sigmf', {'b': 0, 'c': 0.25}], ['sigmf', {'b': 0, 'c': -0.25}], ['sigmf', {'b': 0, 'c': 0.5}], ['sigmf', {'b': 0, 'c': -0.5}]]]

mf = [[['sigmf', {'b': 0, 'c': 0.1}], ['sigmf', {'b': 0, 'c': -0.1}]],
      [['sigmf', {'b': 0, 'c': 0.1}], ['sigmf', {'b': 0, 'c': -0.1}]],
      [['sigmf', {'b': 0, 'c': 0.1}], ['sigmf', {'b': 0, 'c': -0.1}]]]

# mf = [[['sigmf', {'b': -0.2, 'c': 0.1}], ['sigmf', {'b': -0.1, 'c': 0.1}], ['sigmf', {'b': 0.1, 'c': 0.1}], ['sigmf', {'b': 0.2, 'c': 0.1}]],
#       [['sigmf', {'b': -0.2, 'c': 0.1}], ['sigmf', {'b': -0.1, 'c': 0.1}], ['sigmf', {'b': 0.1, 'c': 0.1}], ['sigmf', {'b': 0.2, 'c': 0.1}]],
#       [['sigmf', {'b': -0.2, 'c': 0.1}], ['sigmf', {'b': -0.1, 'c': 0.1}], ['sigmf', {'b': 0.1, 'c': 0.1}], ['sigmf', {'b': 0.2, 'c': 0.1}]]]


print "Starting membership function"
mfc = membership.membershipfunction.MemFuncs(mf)

print "Finished membership function, starting ANFIS"
anf = anfis.ANFIS(X, Y, mfc)

anf.plotMF(x, 0)

t = time.time()

print "Finished ANFIS, starting training Hybrid"
anf.trainHybridJangOffLine(epochs=10)

with open('fuzzy_test_sigmf_175_185.pkl', 'wb') as f:
    pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

print time.time() - t

# var = numpy.array([[3, 4], [3, 4]])
var = numpy.array([[0,0,-207]])

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
