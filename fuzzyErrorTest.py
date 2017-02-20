from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import timeit
import cPickle as pickle

# ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
test_xyz = numpy.loadtxt("test_table_176_184.csv", delimiter=',', usecols=[0,1,2])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
test_angles = numpy.loadtxt("test_table_176_184.csv", delimiter=',', usecols=[3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])


with open('fuzzy_test_bell.pkl', 'rb') as f:
    anf = pickle.load(f)

anf.plotErrors()
anf.plotResults()

total_error = 0
average_error = []

for row_num in range(len(test_xyz)):
    input_val = test_xyz[row_num]
    input_val = numpy.array([input_val])
    # print input_val

    predicted_output = anfis.predict(anf, input_val)

    # print "Expected Output: ", test_angles[row_num]
    # print "Predicted Output: ", predicted_output

    percent_error = numpy.mean(abs(test_angles[row_num] - predicted_output) / test_angles[row_num])
    # print "Percent Error: ", percent_error

    total_error = total_error + percent_error

    average_error.append(percent_error)

    print row_num, len(test_xyz)

print "Total Error: ", total_error
print "Average Error: ", numpy.mean(average_error)

# input_val = numpy.array([[2.8421709430404007e-14,-7.1054273576010019e-15,-171.89997335979379]])
# output_val = numpy.array([150.0,150.0,150.0])
#
# start = timeit.timeit()
# # print the predicted value based on the trained set
# anfis_ans = anfis.predict(anf, input_val)
# print "The answer is", anfis_ans
#
# end = timeit.timeit()
# print "Time: ", end - start
#
# mse = ((output_val - anfis_ans) ** 2).mean(axis=None)
# print "Error: ", mse * 100
