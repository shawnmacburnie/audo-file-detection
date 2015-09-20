__author__ = 'shawn'
from rbm.RBM import *
from rbm.logistic_sgd import load_data
x = T.matrix('x')

def train_rbm(rbm, hidden_rbm, train_set_x, learning_rate=0.1, training_epochs=10,
              batch_size=20, n_hidden=500, CD_steps=3):
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=CD_steps)
    index = T.lscalar()
    name = 'train_rbm'
    if hidden_rbm == None:
        name += '_hidden'
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name=name
    )
    for epoch in xrange(training_epochs):
        print('Training on ' + str(epoch))
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

    if hidden_rbm != None:
        print("starting next rbm")
        L2_data = to_shared(rbm.sample_h_given_v(train_set_x)[-1].eval())
        train_layer_2(hidden_rbm, None, L2_data,0.1, 1,
                  batch_size, n_hidden, 1)

def train_layer_2(rbm, hidden_rbm, train_set_x, learning_rate=0.1, training_epochs=10,
              batch_size=20, n_hidden=500, CD_steps=3):
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=CD_steps)
    index = T.lscalar()
    name = 'train_rbm'
    if hidden_rbm == None:
        name += '_hidden'
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name=name
    )
    for epoch in xrange(training_epochs):
        print('Training on ' + str(epoch))
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))


def to_shared(data):
    return theano.shared(numpy.asarray(data,dtype=theano.config.floatX))


n_hidden = 500
n_visable = 28 * 28
x = T.matrix('x')
x2 = T.matrix('x2')


rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
rng2 = numpy.random.RandomState(123)
theano_rng2 = RandomStreams(rng.randint(2 ** 30))
rbm = RBM(input=x, n_visible=28 * 28,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

hidden_rbm = RBM(input=x, n_visible=n_hidden,
              n_hidden=n_hidden, numpy_rng=rng2, theano_rng=theano_rng2)

# L2_data = to_shared(rbm.sample_h_given_v(train_set_x)[-1].eval())

train_rbm(rbm, hidden_rbm, train_set_x, learning_rate=0.1, training_epochs=1,
              batch_size=20, n_hidden=500, CD_steps=1)

def get_training_data(file_name, window_size, window_inc):
    '''
    :param file_name: Name of the file you want to load
    :param window_size: size of each window in tics (lines in file)
    :param window_inc: how much you want to jump to get to next window.
    :return: shared variable containing all a matrix of windows to train on.
    '''
    with open(file_name) as f:
        lines = f.readlines().split('\n')
        training_data = []
        index = 0
        running = True
        while True:
            vectors = []
            if index+ window_inc >=len(lines):
                running = False
                vectors = lines [index: len(lines)]
            else:
                vectors = lines[index:index + window_inc]
            window_example = []
            # for vector in vectors:
                #this logic needs to be finished. I will try and work this later tonight.

        return to_shared(training_data)

