#support functions
import sys
import numpy
import logging
import cPickle
from mlp.costs import CECost
from mlp.layers import MLP, Softmax, Sigmoid
import matplotlib.pyplot as plt

from mlp.optimisers import SGDOptimiser, Optimiser#import the optimiser
from mlp.dataset import MNISTDataProvider #import data provider #Ruslan Burakov - s1569105
from mlp.schedulers import LearningRateFixed

rng = numpy.random.RandomState([2015,10,10])
rng_state = rng.get_state()

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#preload data
logger.info("Data Preloading")
train_dp_flat = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
valid_dp_flat = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False)
test_dp_flat = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
logger.info("Data Preloading is over")

def count_num_hidden_units(n_in_uts, n_out_uts, n_total_whts, n_hlrs):
    a = float(n_hlrs - 1)
    b = float(n_in_uts + n_out_uts + n_hlrs)
    c = float(n_out_uts - n_total_whts)
    answer = None
    if n_hlrs == 1:
        return int(-c/b)

    opd = (b * b - 4 * a * c)**0.5
    #only x1 because min_point = -b/2a and b/a always > 0 so always min_point < 0
    x1 = (-b + opd) / (2 * a)
    return x1

def count_total_num_weights(n_in_uts, n_out_uts, n_whts_per_hlrs, n_hlrs):
    a = float(n_hlrs - 1)
    b = float(n_in_uts + n_out_uts + n_hlrs)
    c = float(n_out_uts)
    return  n_whts_per_hlrs * n_whts_per_hlrs * a + n_whts_per_hlrs * b + c

def create_one_hid_model():
    """
    In CW1 with default parameters (learning rate was 0.5) this
    model achieved 97.5% accuracy and it takes only 3 seconds per epoch.
    That is why I think it will suit as petri dish for many tasks.
    """
    cost = CECost()
    model = MLP(cost=cost)
    model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
    model.add_layer(Softmax(idim=100, odim=10, rng=rng))
    return model

def create_many_hid_models(n_hlrs,
                           total_weights,
                           layer_fun = lambda idim, odim, rng: Sigmoid(idim=idim, odim=odim, rng=rng)):
    n_whts_per_hlrs = int(count_num_hidden_units(784, 10, total_weights, n_hlrs))
    cost = CECost()
    model = MLP(cost=cost)
    model.add_layer(layer_fun(idim=784, odim=n_whts_per_hlrs, rng=rng))
    for hlr in range(1, n_hlrs):
        model.add_layer(Sigmoid(idim=n_whts_per_hlrs, odim=n_whts_per_hlrs, rng=rng))
    model.add_layer(Softmax(idim=n_whts_per_hlrs, odim=10, rng=rng))
    return model

def test_model(model, label="no_name", test_dp=test_dp_flat):
    logger.info('Testing the model {0} on test set:'.format(label))
    test_dp_flat.reset()
    optimiser = Optimiser()
    cost, accuracy = optimiser.validate(model, test_dp)
    logger.info('MNIST test set accuracy is %.2f %% (cost is %.3f)'%(accuracy * 100., cost))
    return cost, accuracy
    
def learn(**kwargs):
    #to not interrupt other tasks
    try:
        if "model" not in kwargs:
            logger.info("Don't found a model. Return [(0, 0)], [(0, 0)]")
            print "Don't found a model. Return [(0, 0)], [(0, 0)]"
            return [(0, 0)], [(0, 0)]
        #default parameters
        model = kwargs["model"]
        label = kwargs["label"] if "label" in kwargs else "no_name"
        max_epochs = kwargs["max_epochs"] if "max_epochs" in kwargs else 30
        lr_rate = kwargs["lr_rate"] if "lr_rate" in kwargs  else 0.1
        lr_scheduler = kwargs["lr_scheduler"] if "lr_scheduler" in kwargs else LearningRateFixed(lr_rate, max_epochs)
        optimiser = kwargs["optimiser"] if "optimiser" in kwargs else None
        valid_dp = kwargs["valid_dp"] if "valid_dp" in kwargs else valid_dp_flat
        train_dp = kwargs["train_dp"] if "train_dp" in kwargs else train_dp_flat
        save = kwargs["save"] if "save" in kwargs else False

        if optimiser is None:
            optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

        logger.info('Reinitialising data providers...')
        #randomize
        valid_dp.reset()
        train_dp.reset()
        logger.info('Training started {0} ...'.format(label))

        tr_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)
        if save:
            print "saving data for " + label
            logger.info("saving data for " + label)
            #saving for future use
            with open(label + '_model.pkl','wb') as f:
                cPickle.dump(model, f)
            with open(label + '_train_stats.pkl','wb') as f:
                    cPickle.dump(tr_stats, f)
            with open(label + '_valid_stats.pkl','wb') as f:
                cPickle.dump(valid_stats, f)

        #training, validation sets stats
        return tr_stats, valid_stats

    except Exception as e:
        print str(e)
        logger.info(str(e))
        logger.info("Return [(0, 0)], [(0, 0)]")
        print "Return [(0, 0)], [(0, 0)]"
        return [(0, 0)], [(0, 0)]

#(in order not to recalculate it each time we make changes in plot functions)
def get_models_statistics(tasks):
    #num_models x 2 {train, validation} x epoches x 2 {error_cost, accuracy}
    return numpy.array([learn(**task) for task in tasks])

#all of them return #-> num_models x epoches
def get_train_accuracies(models_statistics):
    return models_statistics[:, 0, :, 1]

def get_train_error_costs(models_statistics):
    return models_statistics[:, 0, :, 0]

def get_valid_accuracies(models_statistics):
    return models_statistics[:, 1, :, 1]

def get_valid_error_costs(models_statistics):
    return models_statistics[:, 1, :, 0]



def plot_epoch_dynamic(values_list, labels, y_label, loc = 1):
    lines = []
    for label, values in zip(labels, values_list):
        line, = plt.plot(values, label=label)
        lines.append(line)

    num_epoches = len(values_list[0])
    plt.xlabel("epoch")
    plt.xticks(range(0, num_epoches, 2))
    plt.ylabel(y_label)
    plt.legend(lines, loc=loc)

def plot_error_rate(tr_accrs, vd_accrs, labels, figsize=(12,12)):
    plt.figure(1, figsize=figsize)
    plot_epoch_dynamic([(1 - ta) * 100 for ta in tr_accrs], labels, "training error rate")
    plt.figure(2, figsize=figsize)
    plot_epoch_dynamic([(1 - va) * 100 for va in vd_accrs], labels, "validation error rate")
    plt.show()

def print_test_error_rate(test_accrs, learning_rates, padding=20):
    row_format = ("{:>" + str(padding) + "}") * 2
    header = row_format.format("model", "error rate")
    table = "\n".join([ row_format.format(lr, (1 - test_accr) * 100)\
                       for lr, test_accr in zip(learning_rates, test_accrs)])
    print header
    print table
