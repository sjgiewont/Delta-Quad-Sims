from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy as np
import time
import cPickle as pickle
from fuzzyErrorTest import fuzzy_error_test
import sys
import logging
import smtplib
from email.mime.text import MIMEText

# Run this from the cmd line, with arguments including the training CSV and then the number of epochs
# python neuroFuzzyControl_7.py name_of_csv_file.csv number_of _epochs
#
# Will log all events to a .log file titled with the CSV filename and number of epochs used
# will also store the final pickle file with the CSV filename and the number of epochs used


training_data_csv = sys.argv[1]
epoch_num = sys.argv[2]

while int(epoch_num) < 50:

    log_filename = []
    log_filename.append('fuzzy_log_')
    log_filename.append(training_data_csv[:-4])
    log_filename.append('_epoch_')
    log_filename.append(epoch_num)
    log_filename.append('.log')
    log_filename = ''.join(log_filename)



    # msg = MIMEText(log_filename)
    # msg['Subject'] = 'ANFIS Pickle File Done'
    # msg['From'] = 'stevyg11@gmail.com'
    # msg['To'] = 'stevyg11@gmail.com'
    # print msg
    # # Send the message via our own SMTP server, but don't include the
    # # envelope header.
    # s = smtplib.SMTP('myserver')
    # s.sendmail('stevyg11@gmail.com', ['stevyg11@gmail.com'], msg.as_string())
    # s.quit()

    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

    try:
        logging.info('Logging Data for neuroFuzzyControl_7')
        logging.info('Training CSV = %s', training_data_csv)
        logging.info('Epochs = %s', epoch_num)

        logging.info('Started Loading Training Data')
        # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        # ts = numpy.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
        ts = np.loadtxt(training_data_csv, delimiter=',', usecols=[0,1,2,3,4,5])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])

        # using coord input, theta output
        # X = ts[:,0:3]
        # Y = ts[:,3:7]

        X = ts[:,0:3]
        Y = ts[:,3:6]

        logging.info('Finished Loading Training Data')
        logging.info(X)
        logging.info(Y)
        print X
        print Y

        x = ts[:, 0]
        y = ts[:, 1]
        z = ts[:, 2]

        x_step = (np.max(x) - np.min(x)) / 4
        x_start = np.min(x) + (x_step / 2)
        x_sigma = x_step / 2

        y_step = (np.max(y) - np.min(y)) / 4
        y_start = np.min(y) + (y_step / 2)
        y_sigma = y_step / 2

        z_step = (np.max(z) - np.min(z)) / 4
        z_start = np.min(z) + (z_step / 2)
        z_sigma = z_step / 2

        x_mu_1 = x_start
        x_mu_2 = x_start + x_step
        x_mu_3 = x_start + 2 * x_step
        x_mu_4 = x_start + 3 * x_step
        x_width_1 = 10*x_sigma
        x_slope_1 = 1

        y_mu_1 = y_start
        y_mu_2 = y_start + y_step
        y_mu_3 = y_start + 2 * y_step
        y_mu_4 = y_start + 3 * y_step
        y_width_1 = 10*y_sigma
        y_slope_1 = 1

        z_mu_1 = z_start
        z_mu_2 = z_start + z_step
        z_mu_3 = z_start + 2 * z_step
        z_mu_4 = z_start + 3 * z_step
        z_width_1 = 10*z_sigma
        z_slope_1 = 1

        mf = [[['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_1}],
               ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_2}],
               ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_3}],
               ['gbellmf', {'a': x_width_1, 'b': x_slope_1, 'c': x_mu_4}]],
              [['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_1}],
               ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_2}],
               ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_3}],
               ['gbellmf', {'a': y_width_1, 'b': y_slope_1, 'c': y_mu_4}]],
              [['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_1}],
               ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_2}],
               ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_3}],
               ['gbellmf', {'a': z_width_1, 'b': z_slope_1, 'c': z_mu_4}]]]

        x = np.linspace(np.min(x)+np.min(x)/2, np.max(x)+np.max(x)/2, 100)
        z_linspace = np.linspace(np.min(z)-z_sigma, np.max(z)+z_sigma, 100)

        print "Starting membership function"
        logging.info("Starting membership function")
        mfc = membership.membershipfunction.MemFuncs(mf)

        print "Finished membership function, starting ANFIS"
        logging.info("Finished membership function, starting ANFIS")
        anf = anfis.ANFIS(X, Y, mfc)

        t = time.time()

        print "Finished ANFIS, starting training Hybrid"
        logging.info("Finished ANFIS, starting training Hybrid")
        anf.trainHybridJangOffLine(epochs=int(epoch_num))

        pickle_filename = []
        pickle_filename.append(log_filename[:-4])
        pickle_filename.append('.pkl')
        pickle_filename = ''.join(pickle_filename)

        with open(pickle_filename, 'wb') as f:
            pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

        logging.info("Wrote the PICKLE FILE: %s", pickle_filename)

        print time.time() - t

        total_error, average_error = fuzzy_error_test(anf, "test_table_176_184.csv")

        logging.info("The total error is: %f", total_error)
        logging.info("The average error is: %f", average_error)
        print "The total error is: ", total_error
        print "The average error is: ", average_error

        # var = numpy.array([[3, 4], [3, 4]])
        var = np.array([[0,0,-207]])

        print "input", var

        # print the predicted value based on the trained set
        print "The answer is", anfis.predict(anf, var)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("favorfulxyz@gmail.com", "myfavorfulpassword1g")

        msg = pickle_filename
        server.sendmail("favorfulxyz@gmail.com", "stevyg11@gmail.com", msg)
        server.quit()

        epoch_num = epoch_num + 5

    except:
        logging.exception('An exception has occured!!')
        raise
