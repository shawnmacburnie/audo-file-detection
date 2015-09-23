__author__ = 'shawn'
from rbm.RBM import *
from data_suite import get_training_data as data_gen
from training import train_rbm
from rbm import create_rbm



n_hidden = 800
x = T.matrix('x')
n_visable, train_set_x = data_gen.get_training_data('test-sageev-bach-1_extracted.txt_net_data.txt',16,8)

rbm = create_rbm.create(n_visable,n_hidden, x)
hidden_rbm = create_rbm.create(n_hidden, n_hidden, x)

train_rbm.train_two_layer_rbm(rbm, hidden_rbm, train_set_x, x,learning_rate=0.01, training_epochs=25,
              batch_size=4, n_hidden=n_hidden, CD_steps=15)



