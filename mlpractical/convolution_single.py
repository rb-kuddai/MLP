import numpy
import logging
import sys
import os
import cPickle

os.environ['OMP_NUM_THREADS'] = "3"
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


from mlp.layers import MLP, Sigmoid, Linear, Softmax #import required layer types
from mlp.conv import ConvLinear, ConvRelu, ConvSigmoid
from mlp.maxpooling import ConvMaxPool2D
from mlp.optimisers import SGDOptimiser, Optimiser#import the optimiser
from mlp.dataset import MNISTDataProvider #import data provider #Ruslan Burakov - s1569105
from mlp.costs import CECost, MSECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed

rng = numpy.random.RandomState([2015,10,10])

# define the model structure, here just one linear layer
# and mean square error cost
tsk8_1_cost = CECost()
tsk8_1_model = MLP(cost=tsk8_1_cost)
"""
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my_conv_fwd,
                 conv_bck=my_conv_bck,
                 conv_grad=my_conv_grad)
"""
tsk8_1_model.add_layer(ConvSigmoid(num_inp_feat_maps=1,
                            num_out_feat_maps=1,
                            image_shape=(28,28),
                            kernel_shape=(5, 5),
                            stride=(1,1),
                            rng=rng))

tsk8_1_model.add_layer(ConvMaxPool2D(num_feat_maps=1,
                              conv_shape=(24, 24),
                              pool_shape=(2, 2),
                              pool_stride=(2, 2) ))
#idim, odim,
tsk8_1_model.add_layer(Sigmoid(idim=12*12, odim=100, rng=rng))
tsk8_1_model.add_layer(Softmax(idim=100, odim=10, rng=rng))
#one can stack more layers here

# define the optimiser, here stochasitc gradient descent
# with fixed learning rate and max_epochs as stopping criterion
lr_scheduler = LearningRateFixed(learning_rate=0.1, max_epochs=30)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

logger.info('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True, conv_reshape=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False, conv_reshape=True)

logger.info('Training started...')
tsk8_1_tr_stats, tsk8_1_valid_stats = optimiser.train(tsk8_1_model, train_dp, valid_dp)

logger.info('Testing the model on test set:')
test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False, conv_reshape=True)
tsk8_1_cost, tsk8_1_accuracy = optimiser.validate(tsk8_1_model, test_dp)
logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)'%(tsk8_1_accuracy*100., tsk8_1_cost))

#saving for future use
with open('tsk8_1_model.pkl','wb') as f:
    cPickle.dump(tsk8_1_model, f)
#saving for future use
with open('tsk8_1_tr_stats.pkl','wb') as f:
    cPickle.dump(tsk8_1_tr_stats, f)
#saving for future use
with open('tsk8_1_valid_stats.pkl','wb') as f:
    cPickle.dump(tsk8_1_valid_stats, f)