__author__ = 'shawn'
from rbm.RBM import *
def create(n_visable, n_hidden, x):
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rng2 = numpy.random.RandomState(123)
    theano_rng2 = RandomStreams(rng.randint(2 ** 30))

    return RBM(input=x, n_visible=n_visable,
              n_hidden=n_hidden, numpy_rng=rng2, theano_rng=theano_rng2)