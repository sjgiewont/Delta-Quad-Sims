from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import timeit
import cPickle as pickle

with open('fuzzycontrol_170_175.pkl', 'rb') as f:
    anf = pickle.load(f)

# var = numpy.array([[0.9,-1,-135.1]])
input_val = numpy.array([[2.8421709430404007e-14,-7.1054273576010019e-15,-171.89997335979379]])
output_val = numpy.array([150.0,150.0,150.0])

print "input", input_val

# print "the length is", len(var[:,0])
#
# var = numpy.asanyarray(var)
# print "the shape", var.shape
# print "the shape index", var.shape[0]
# print "the shape index", var.shape[1]
start = timeit.timeit()
# print the predicted value based on the trained set
anfis_ans = anfis.predict(anf, input_val)
print "The answer is", anfis_ans

end = timeit.timeit()
print "Time: ", end - start

mse = ((output_val - anfis_ans) ** 2).mean(axis=None)
print "Error: ", mse * 100

