__author__ = 'shawn'
from rbm.RBM import *
from rbm.logistic_sgd import load_data

n_hidden = 500
n_visable = 28 * 28
x = T.matrix('x')
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
rbm = RBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]




# single = theano.shared(
#         numpy.asarray(
#             test_set_x.get_value(borrow=True)[0],
#             dtype=theano.config.floatX
#         )
#     )
# l = rbm.sample_h_given_v(single)[-1]
# single = theano.shared(
#         numpy.asarray(
#             l.eval(),
#             dtype=theano.config.floatX
#         )
#     )
# l = rbm.sample_v_given_h(single)[-1]
# print(len(l.eval()))

