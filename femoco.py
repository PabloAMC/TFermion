import h5py
import numpy

hfdump = h5py.File('integrals/eri_li.h5','r')
#hfdump = h5py.File('integrals/eri_reiher.h5','r') #alternative
eri = hfdump.get('eri')[()]
H1 = hfdump.get('h0')[()]
hfdump.close()


# compute one-body
T = H1 - 0.5 * numpy.einsum("pqqs->ps", eri) + numpy.einsum("pqrr->pq", eri)

