'''
This script evaluates the ANFIS network stored in a PICKLE file. It records the time it takes to evaluate, and the error
between the expected and the predicted value.
'''

from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import timeit
import cPickle as pickle

# choose the anfis pickle file
with open('fuzzy_test_gauss.pkl', 'rb') as f:
    anf = pickle.load(f)

# set the input value
input_val = numpy.array([[0, 0, -207.70890486488082]])

# set what the expected output should be
output_val = numpy.array([175,175,175])

print "input", input_val

# print "the length is", len(var[:,0])
#
# var = numpy.asanyarray(var)
# print "the shape", var.shape
# print "the shape index", var.shape[0]
# print "the shape index", var.shape[1]

# start the timer
start = timeit.timeit()

# print the predicted value based on the trained set
anfis_ans = anfis.predict(anf, input_val)
print "The answer is", anfis_ans

# stop the timer
end = timeit.timeit()
print "Time: ", end - start

# calculate the error
mse = ((output_val - anfis_ans) ** 2).mean(axis=None)
print "Error: ", mse * 100

