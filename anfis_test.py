'''-----------------------------------------------------------
This is a brief test of ANFIS using a 2-input, 1-output training set. 

This will train the network, print out the status as it is functioning. 

Will plot the errors and the predicted values. 
-----------------------------------------------------------'''

import membership
import numpy
from anfis import *

# import the training set
ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])

# take the input values to be columns 1 and 2
X = ts[:,0:2]

# take the predicted values to be from column 3
Y = ts[:,2]

# define the parameters of the membership function with multiple gaussian membership curves
mf = [[['gaussmf',{'mean':-11.,'sigma':5.}],['gaussmf',{'mean':-8.,'sigma':5.}],['gaussmf',{'mean':-14.,'sigma':20.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':-10.,'sigma':20.}],['gaussmf',{'mean':-20.,'sigma':11.}],['gaussmf',{'mean':-9.,'sigma':30.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]

# create the membership function from the parameters
mfc = membership.membershipfunction.MemFuncs(mf)

# create the ANFIS object, defining the input, output and memebership function to use
anf = anfis.ANFIS(X, Y, mfc)

# train ANFIS to learn the correlation between the inputs (X) and output (Y)
anf.trainHybridJangOffLine(epochs=10)

# print the consequents and fitted values. Make sure you are getting good results.
print round(anf.consequents[-1][0],6)
print round(anf.consequents[-2][0],6)
print round(anf.fittedValues[8][0],6)
print len(anf.fittedValues)
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print 'test is good'

# plot a graph of the overall error along with the predicted values.
anf.plotErrors()
anf.plotResults()
