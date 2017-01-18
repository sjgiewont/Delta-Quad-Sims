from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import time
import cPickle as pickle

start = -170
step = 1

end_range = start - step

while -300 < end_range:

    filename = "test_140_220_half_%d_%d.csv" % (abs(start), abs(end_range))

    # ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
    ts = numpy.loadtxt(filename, delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

    # using coord input, theta output
    X = ts[:,0:3]
    Y = ts[:,3:7]

    x = ts[:,0]
    y = ts[:,1]

    x_step = (numpy.max(x) - numpy.min(x)) / 4
    x_start = numpy.min(x) + (x_step/2)
    x_sigma = x_step / 2

    y_step = (numpy.max(y) - numpy.min(y)) / 4
    y_start = numpy.min(y) + (y_step/2)
    y_sigma = y_step / 2



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

    mf = [[['gaussmf',{'mean':x_start,'sigma':x_sigma}],['gaussmf',{'mean':x_start + x_step,'sigma':x_sigma}],['gaussmf',{'mean':x_start + 2 * x_step,'sigma':x_sigma}],['gaussmf',{'mean':x_start + 3 * x_step,'sigma':x_sigma}]],
                [['gaussmf',{'mean':y_start,'sigma':y_sigma}],['gaussmf',{'mean':y_start + y_step,'sigma':y_sigma}],['gaussmf',{'mean':y_start + 2 * y_step,'sigma':y_sigma}],['gaussmf',{'mean':y_start + 3 * y_step,'sigma':y_sigma}]],
                [['gaussmf', {'mean': end_range, 'sigma': 1.}], ['gaussmf', {'mean': end_range + 2., 'sigma': 1.}], ['gaussmf',{'mean': end_range + 3.,'sigma': 1.}], ['gaussmf', {'mean': start, 'sigma': 1}]]]


    print "Starting membership function"
    mfc = membership.membershipfunction.MemFuncs(mf)

    print "Finished membership function, starting ANFIS"
    anf = anfis.ANFIS(X, Y, mfc)

    t = time.time()

    print "Finished ANFIS, starting training Hybrid"
    anf.trainHybridJangOffLine(epochs=5)

    pickle_filename = "fuzzycontrol_%d_%d.pkl" % (abs(start), abs(end_range))

    with open(pickle_filename, 'wb') as f:
        pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

    print time.time() - t

    start -= step
    end_range -= step


