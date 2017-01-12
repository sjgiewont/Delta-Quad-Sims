from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import time
import cPickle as pickle

# ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
ts = numpy.loadtxt("fuzzy_train_table.csv", delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

X = ts[:,0:3]
Y = ts[:,3:7]
# Y = ts[:,3]

print X
print Y

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

mf = [[['gaussmf',{'mean':-20.,'sigma':10.}],['gaussmf',{'mean':0.,'sigma':10.}],['gaussmf',{'mean':20.,'sigma':10.}]],
            [['gaussmf',{'mean':-20.,'sigma':10.}],['gaussmf',{'mean':0.,'sigma':10.}],['gaussmf',{'mean':20.,'sigma':10.}]],
            [['gaussmf', {'mean': -100., 'sigma': 5.}], ['gaussmf', {'mean': -110., 'sigma': 5.}], ['gaussmf',{'mean': -120.,'sigma': 5.}]]]

print "Starting membership function"
mfc = membership.membershipfunction.MemFuncs(mf)

print "Finished membership function, starting ANFIS"
anf = anfis.ANFIS(X, Y, mfc)

t = time.time()

print "Finished ANFIS, starting training Hybrid"
anf.trainHybridJangOffLine(epochs=5)

with open('fuzzycontrol.pkl', 'wb') as f:
    pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

print time.time() - t

# var = numpy.array([[3, 4], [3, 4]])
var = numpy.array([[1, 1, 1]])

print "input", var

# print "the length is", len(var[:,0])
#
# var = numpy.asanyarray(var)
# print "the shape", var.shape
# print "the shape index", var.shape[0]
# print "the shape index", var.shape[1]

# print the predicted value based on the trained set
print "The answer is", anfis.predict(anf, var)

