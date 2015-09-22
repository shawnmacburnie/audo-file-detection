__author__ = 'shawn'
import theano
import numpy
import theano.tensor as T
from data_suite import get_training_data as data_gen
xrange = range

def train_two_layer_rbm(rbm, hidden_rbm, train_set_x, x, learning_rate=0.1, training_epochs=10,
              batch_size=20, n_hidden=500, CD_steps=3, ):
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
        L2_data = data_gen.to_shared(rbm.sample_h_given_v(train_set_x)[-1].eval())
        train_layer_2(hidden_rbm, None, L2_data, x, learning_rate, training_epochs,
                  batch_size, n_hidden, CD_steps)

def train_layer_2(rbm, hidden_rbm, train_set_x, x, learning_rate=0.1, training_epochs=10,
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