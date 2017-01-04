from anfis import *
import membership #import membershipfunction, mfDerivs
import numpy

ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
X = ts[:,0:2]
Y = ts[:,2]
X = numpy.matrix('1 2; 3 4; 5 6')
Y = numpy.array([6, 3, 1])



mf = [[['gaussmf',{'mean':-11.,'sigma':5.}],['gaussmf',{'mean':-8.,'sigma':5.}],['gaussmf',{'mean':-14.,'sigma':20.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':-10.,'sigma':20.}],['gaussmf',{'mean':-20.,'sigma':11.}],['gaussmf',{'mean':-9.,'sigma':30.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]


mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=10)

print round(anf.fittedValues[0][0],6)

# var = numpy.array([[3, 4], [3, 4]])
var = numpy.array([[3, 4]])

print var

print "the length is", len(var[:,0])

var = numpy.asanyarray(var)
print "the shape", var.shape
print "the shape index", var.shape[0]
print "the shape index", var.shape[1]

# print the predicted value based on the trained set
print anfis.predict(anf, var)

