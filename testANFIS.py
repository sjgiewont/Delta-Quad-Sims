from fuzzyErrorTest import *
import glob, os, sys
import logging

validation_data_csv = sys.argv[1]

log_filename = []
log_filename.append('anfis_test_')
log_filename.append(validation_data_csv[:-4])
log_filename.append('.log')
log_filename = ''.join(log_filename)

logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

try:
    logging.info('Logging Data for testANFIS')
    logging.info('Training CSV = %s', validation_data_csv)

    # test_table_csv = "test_table_176_184.csv"

    for pickle_file in glob.glob("*.pkl"):
        print(pickle_file)
        logging.info("PICKLE FILE: %s", pickle_file)

        with open(pickle_file, 'rb') as f:
            anf = pickle.load(f)
        total_error, average_error = fuzzy_error_test(anf, validation_data_csv)
        logging.info("Total Error: %s", total_error)
        logging.info("Total Error: %s", average_error)


except:
    logging.exception('An exception has occured!!')
    raise
