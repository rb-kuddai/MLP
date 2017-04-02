# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import logging


class LearningRateScheduler(object):
    """
    Define an interface for determining learning rates
    """
    def __init__(self, max_epochs=100):
        self.epoch = 0
        self.max_epochs = max_epochs

    def get_rate(self):
        raise NotImplementedError()

    def get_next_rate(self, current_accuracy=None):
        self.epoch += 1


class LearningRateList(LearningRateScheduler):
    def __init__(self, learning_rates_list, max_epochs):

        super(LearningRateList, self).__init__(max_epochs)

        assert isinstance(learning_rates_list, list), (
            "The learning_rates_list argument expected"
            " to be of type list, got %s" % type(learning_rates_list)
        )
        self.lr_list = learning_rates_list
        
    def get_rate(self):
        if self.epoch < len(self.lr_list):
            return self.lr_list[self.epoch]
        return 0.0
    
    def get_next_rate(self, current_accuracy=None):
        super(LearningRateList, self).get_next_rate(current_accuracy=None)
        return self.get_rate()


class LearningRateFixed(LearningRateList):

    def __init__(self, learning_rate, max_epochs):
        assert learning_rate > 0, (
            "learning rate expected to be > 0, got %f" % learning_rate
        )
        super(LearningRateFixed, self).__init__([learning_rate], max_epochs)

    def get_rate(self):
        if self.epoch < self.max_epochs:
            return self.lr_list[0]
        return 0.0

    def get_next_rate(self, current_accuracy=None):
        super(LearningRateFixed, self).get_next_rate(current_accuracy=None)
        return self.get_rate()

class LearningRateExponential(LearningRateList):
    def __init__(self, initial_learning_rate, max_epochs, r):
        assert initial_learning_rate > 0, (
            "initial learning rate expected to be > 0, got %f" % initial_learning_rate
        )
        assert r > 0, (
            "r expected to be > 0, got %f" % r
        )

        import numpy as np
        tt = np.arange(0, max_epochs)
        yy = initial_learning_rate * np.exp(-tt / r)
        super(LearningRateExponential, self).__init__(list(yy), max_epochs)

class LearningRateReciprocal(LearningRateList):
    def __init__(self, initial_learning_rate, max_epochs, r, c):
        assert initial_learning_rate > 0, (
            "initial learning rate expected to be > 0, got %f" % initial_learning_rate
        )
        assert r > 0, (
            "r to be > 0, got %f" % r
        )
        assert c >= 0, (
            "c expected to be > 0, got %f" % c
        )
        import numpy as np
        tt = np.arange(0, max_epochs)
        yy = initial_learning_rate * (1 + tt / r) ** (-c)
        super(LearningRateReciprocal, self).__init__(list(yy), max_epochs)



class LearningRateNewBob(LearningRateScheduler):
    """
    newbob learning rate schedule.
    
    Fixed learning rate until validation set stops improving then exponential
    decay.
    """
    
    def __init__(self, start_rate, scale_by=.5, max_epochs=99,
                 min_derror_ramp_start=.5, min_derror_stop=.5, init_error=100.0,
                 patience=0, zero_rate=None, ramping=False):
        """
        :type start_rate: float
        :param start_rate: 
        
        :type scale_by: float
        :param scale_by: 
        
        :type max_epochs: int
        :param max_epochs: 
        
        :type min_error_start: float
        :param min_error_start: 
        
        :type min_error_stop: float
        :param min_error_stop: 
        
        :type init_error: float
        :param init_error: 
        """
        self.start_rate = start_rate
        self.init_error = init_error
        self.init_patience = patience
        
        self.rate = start_rate
        self.scale_by = scale_by
        self.max_epochs = max_epochs
        self.min_derror_ramp_start = min_derror_ramp_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error
        
        self.epoch = 1
        self.ramping = ramping
        self.patience = patience
        self.zero_rate = zero_rate
        
    def reset(self):
        self.rate = self.start_rate
        self.lowest_error = self.init_error
        self.epoch = 1
        self.ramping = False
        self.patience = self.init_patience
    
    def get_rate(self):
        if (self.epoch==1 and self.zero_rate!=None):
            return self.zero_rate
        return self.rate  
    
    def get_next_rate(self, current_accuracy):
        """
        :type current_accuracy: float
        :param current_accuracy: current proportion correctly classified
        
        """
        
        current_error = 1. - current_accuracy
        diff_error = 0.0
        
        if ( (self.max_epochs > 10000) or (self.epoch >= self.max_epochs) ):
            #logging.debug('Setting rate to 0.0. max_epochs or epoch>=max_epochs')
            self.rate = 0.0
        else:
            diff_error = self.lowest_error - current_error
            
            if (current_error < self.lowest_error):
                self.lowest_error = current_error
    
            if (self.ramping):
                if (diff_error < self.min_derror_stop):
                    if (self.patience > 0):
                        #logging.debug('Patience decreased to %f' % self.patience)
                        self.patience -= 1
                        self.rate *= self.scale_by
                    else:
                        #logging.debug('diff_error (%f) < min_derror_stop (%f)' % (diff_error, self.min_derror_stop))
                        self.rate = 0.0
                else:
                    self.rate *= self.scale_by
            else:
                if (diff_error < self.min_derror_ramp_start):
                    #logging.debug('Start ramping.')
                    self.ramping = True
                    self.rate *= self.scale_by
            
            self.epoch += 1
    
        return self.rate


class DropoutFixed(LearningRateList):

    def __init__(self, p_inp_keep, p_hid_keep):
        assert 0 < p_inp_keep <= 1 and 0 < p_hid_keep <= 1, (
            "Dropout 'keep' probabilites are suppose to be in (0, 1] range"
        )
        super(DropoutFixed, self).__init__([(p_inp_keep, p_hid_keep)], max_epochs=999)

    def get_rate(self):
        return self.lr_list[0]

    def get_next_rate(self, current_accuracy=None):
        return self.get_rate()

class DropoutFixed(LearningRateList):

    def __init__(self, p_inp_keep, p_hid_keep):
        assert 0 < p_inp_keep <= 1 and 0 < p_hid_keep <= 1, (
            "Dropout 'keep' probabilites are suppose to be in (0, 1] range"
        )
        super(DropoutFixed, self).__init__([(p_inp_keep, p_hid_keep)], max_epochs=999)

    def get_rate(self):
        return self.lr_list[0]

    def get_next_rate(self, current_accuracy=None):
        return self.get_rate()

class DropoutAnnealed(LearningRateScheduler):

    def __init__(self, p_inp_keep0, p_hid_keep0, inc_amnt, lr_scheduler):
        """
        In the implementation of optimisers dp_scheduler is never called
        So I wasn't sure where I have to take the current value of the epoch, and
        I didn't want to store it in two places and sync them.
        That is why I pass lr_scheduler as parameter
        """
        assert 0 < p_inp_keep0 <= 1 and 0 < p_hid_keep0 <= 1, (
            "Dropout 'keep' probabilites are suppose to be in (0, 1] range"
        )
        assert 0 < inc_amnt <= 1, (
            "increase amount must between 0 and 1"
        )

        super(DropoutAnnealed, self).__init__(max_epochs=999)
        self.p_inp_keep0 = p_inp_keep0
        self.p_hid_keep0 = p_hid_keep0
        self.inc_amnt = inc_amnt
        self.lr_scheduler = lr_scheduler

    def get_rate(self):
        epoch = self.lr_scheduler.epoch
        eps = epoch * self.inc_amnt
        p_inp_keep = min(1, self.p_inp_keep0 + eps)
        p_hid_keep = min(1, self.p_hid_keep0 + eps)
        return (p_inp_keep, p_hid_keep)

    def get_next_rate(self, current_accuracy=None):
        super(DropoutAnnealed, self).get_next_rate(current_accuracy)
        return self.get_rate()
