from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import numpy
import time
import cPickle as pickle

with open('fuzzycontrol.pkl', 'rb') as f:
    anf = pickle.load(f)

t = time.time()

print "Finished ANFIS, starting training Hybrid"
anf.trainHybridJangOffLine(epochs=5)

with open('fuzzycontrol_2.pkl', 'wb') as f:
    pickle.dump(anf, f, pickle.HIGHEST_PROTOCOL)

print "The Error is: ", anf.errors

print time.time() - t