import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import time
srng = RandomStreams()

# Creates numpy floating nuber
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# Currently only works for nxn where shape is n
def init_weights(shape):
    return theano.shared(floatX(np.identity(shape)))

# different activation function that was siggested to use rather then sigmoid
def rectify(X):
    return T.maximum(X, 0.)

# stable softmax, where build-in one is said to have issues
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# defines how to update weights
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w_h, w_h2, w_h3,w_h4,w_h5,w_h6,w_h7,w_h8, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))

    h2 = T.nnet.sigmoid(T.dot(h, w_h2))

    h3 = T.nnet.sigmoid(T.dot(h2, w_h3))

    h4 = T.nnet.sigmoid(T.dot(h3, w_h4))

    h5 = T.nnet.sigmoid(T.dot(h4, w_h5))

    h6 = T.nnet.sigmoid(T.dot(h5, w_h6))

    h7 = T.nnet.sigmoid(T.dot(h6, w_h7))

    h8 = T.nnet.sigmoid(T.dot(h7, w_h8))

    py_x = T.nnet.sigmoid(T.dot(h8, w_o))
    return h, h2, py_x

# genereates my training data and test data to just all be same.
def genData():
    data = []
    for i in range(0,128):
        line = [0] * 128
        line[i] = 1
        data.append(line)
    data.append([1] * 128)
    data = np.asarray(data,dtype=theano.config.floatX)
    return data, data

tr,te = genData()
X = T.fmatrix()
Y = T.fmatrix()
w_h = init_weights(128)
w_h2 = init_weights(128)
w_h3 = init_weights(128)
w_h4 = init_weights(128)
w_h5 = init_weights(128)
w_h6 = init_weights(128)
w_h7 = init_weights(128)
w_h8 = init_weights(128)
w_o = init_weights(128)

h, h2, py_x = model(X, w_h, w_h2, w_h3,w_h4,w_h5,w_h6,w_h7,w_h8, w_o)
# Defines theano function to get max value in the results from sampling
y_x = T.argmax(py_x, axis=1)

# cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
# params = [w_h, w_h2, w_o]
# updates = RMSprop(cost, params, lr=0.001)

# train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

# theano function that prints out the predictions for all X's, if you outputs=y_x
# you will get the maxValue for each X in the results. However if your just want the
# activation from sigmoid, pass outputs=py_x
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

print('training started ...')
startTime = time.time()
for i in range(1):
    # for start, end in zip(range(0, len(tr), 128), range(128, len(tr), 128)):
    #     cost = train(tr[start:end], tr[start:end])

    # prints out the
    print (predict(te))
endTime = time.time()

print('Time Took: ' + str((endTime - startTime)))