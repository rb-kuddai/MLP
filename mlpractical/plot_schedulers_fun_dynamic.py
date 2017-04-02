import numpy as np
import matplotlib.pyplot as plt
from mlp.schedulers import LearningRateExponential
from mlp.schedulers import LearningRateReciprocal

def get_xx_yy(sch):
    epoches, values = [],[]
    while sch.get_rate() != 0:
        epoches.append(sch.epoch)
        values.append(sch.get_rate())
        sch.get_next_rate(None)
    return epoches, values

def reciprocal(lr0, max_epoch, r, cc):
    lines = []
    for c in cc:
        xx, yy = get_xx_yy(LearningRateReciprocal(lr0, max_epoch, r, c))
        line, = plt.plot(xx, yy, label="reciprocal with r {}, c {}".format(r, c))
        lines.append(line)
    return lines

def exponential(lr0, max_epoch, r):
    xx, yy = get_xx_yy(LearningRateExponential(lr0, max_epoch, r))
    line, = plt.plot(xx, yy, label="exponential with r {}".format(r))
    return [line]

def default_run():
    """
    In the previous course work the best speed of
    reaching high accuracy values (above > 90%) was reached
    by using learning rate = 0.5. So it logical to want high values
    of learning rate at the beginning (let's say first 10 epoch) and
    then use small values for final tunning
    From the following plots I choose parameters that
    results in high learning rates at the beginning and low one at the end
    and I test my implementation of LearningRateExponential and LearningRateReciprocal, as well
    """
    initial_learning_rate = 0.7
    epoches = 30
    #proportional to overall batch size
    r = 0.0001 * 50000
    cc =[0.5, 1.0, 2]
    lines = []
    plt.figure(0)
    lines += reciprocal(initial_learning_rate, epoches, r, cc)
    lines += exponential(initial_learning_rate, epoches, r)
    plt.legend(lines, loc=1)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.show()

