'''
Function that calculates the error between all values in a CSV table, and the predicted values using the ANFIS object

Returns the total error and the average error calculated
'''

from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import timeit
import cPickle as pickle


def fuzzy_error_test(anf, test_table_csv):
    # grab the input positions from the test table
    test_xyz = numpy.loadtxt(test_table_csv, delimiter=',', usecols=[0, 1, 2])

    # grab the expected angles
    test_angles = numpy.loadtxt(test_table_csv, delimiter=',', usecols=[3, 4, 5])

    # with open('fuzzy_test_gauss.pkl', 'rb') as f:
    #     anf = pickle.load(f)

    # initialize the total error count
    total_error = 0
    average_error = []

    # loop through each row in the demo table
    for row_num in range(len(test_xyz)):
        # grab the input value for ANFIS
        input_val = test_xyz[row_num]

        # put into a numpy array
        input_val = numpy.array([input_val])

        # evaluate ANFIS and get the predicted output
        predicted_output = anfis.predict(anf, input_val)

        # calculate the percent error between the predicted and expected
        percent_error = numpy.mean(abs(test_angles[row_num] - predicted_output) / test_angles[row_num])

        # add onto the total error
        total_error = total_error + percent_error

        average_error.append(percent_error)

        # print for each row evaluated
        print row_num, len(test_xyz)

    return total_error, numpy.mean(average_error)
