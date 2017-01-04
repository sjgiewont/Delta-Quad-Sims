from anfis import *
import membership #import membershipfunction, mfDerivs
import numpy
import timeit
import cPickle as pickle

with open('fuzzycontrol.pkl', 'rb') as f:
    anf = pickle.load(f)



# var = numpy.array([[3, 4], [3, 4]])
var = numpy.array([[1, 1, 1]])

print "input", var

# print "the length is", len(var[:,0])
#
# var = numpy.asanyarray(var)
# print "the shape", var.shape
# print "the shape index", var.shape[0]
# print "the shape index", var.shape[1]
start = timeit.timeit()
# print the predicted value based on the trained set
print "The answer is", anfis.predict(anf, var)

end = timeit.timeit()
print end - start