
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import cPickle
import gzip
import numpy
import os
import logging
#my addition
import doctest


logger = logging.getLogger(__name__)


class DataProvider(object):
    """
    Data provider defines an interface for our
    generic data-independent readers.
    """
    def __init__(self, batch_size, randomize=True, rng=None):
        """
        :param batch_size: int, specifies the number
               of elements returned at each step
        :param randomize: bool, shuffles examples prior
               to iteration, so they are presented in random
               order for stochastic gradient descent training
        :return:
        """
        self.batch_size = batch_size
        self.randomize = randomize
        self._curr_idx = 0
        self.rng = rng

        if self.rng is None:
            seed=[2015, 10, 1]
            self.rng = numpy.random.RandomState(seed)

    def reset(self):
        """
        Resets the provider to the initial state to
        use in another epoch
        :return: None
        """
        self._curr_idx = 0

    def __randomize(self):
        """
        Data-specific implementation of shuffling mechanism
        :return:
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        This method says an object is iterable.
        """
        return self

    def next(self):
        """
        Data-specific iteration mechanism. Called each step 
        (i.e. each iteration in a loop)
        unitl StopIteration() exception is raised.
        :return:
        """
        raise NotImplementedError()

    def num_examples(self):
        """
        Returns a number of data-points in dataset
        """
        return NotImplementedError()



class MNISTDataProvider(DataProvider):
    """
    The class iterates over MNIST digits dataset, in possibly
    random order.
    """
    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True,
                 rng=None,
                 conv_reshape=False):

        super(MNISTDataProvider, self).\
            __init__(batch_size, randomize, rng)

        assert dset in ['train', 'valid', 'eval'], (
            "Expected dset to be either 'train', "
            "'valid' or 'eval' got %s" % dset
        )
        
        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )
        
        if max_num_batches > 0 and max_num_examples > 0:
            logger.warning("You have specified both 'max_num_batches' and " \
                  "a deprecead 'max_num_examples' arguments. We will " \
                  "use the former over the latter.")

        dset_path = './data/mnist_%s.pkl.gz' % dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        with gzip.open(dset_path) as f:
            x, t = cPickle.load(f)

        self._max_num_batches = max_num_batches
        #max_num_examples arg was provided for backward compatibility
        #but it maps us to the max_num_batches anyway
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size      
            
        self.x = x
        self.t = t
        self.num_classes = 10
        self.conv_reshape = conv_reshape

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MNISTDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)

        if self._rand_idx is not None and self._max_num_batches > 0:
            return self.rng.permutation(self._rand_idx)
        else:
            #the max_to_present secures that random examples
            #are returned from the same pool each time (in case
            #the total num of examples was limited by max_num_batches)
            max_to_present = self.batch_size*self._max_num_batches \
                                if self._max_num_batches > 0 else self.x.shape[0]
            return self.rng.permutation(numpy.arange(0, self.x.shape[0]))[0:max_to_present]

    def next(self):
        #print "self.x.shape[0] {}".format(self.x.shape[0])
        has_enough = (self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches <= (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        rval_x = self.x[range_idx]
        rval_t = self.t[range_idx]

        self._curr_idx += self.batch_size

        if self.conv_reshape:
            rval_x = rval_x.reshape(self.batch_size, 1, 28, 28)

        return rval_x, self.__to_one_of_k(rval_t)

    def num_examples(self):
        return self.x.shape[0]

    def num_examples_presented(self):
        return self._curr_idx + 1

    def __to_one_of_k(self, y):
        rval = numpy.zeros((y.shape[0], self.num_classes), dtype=numpy.float32)
        for i in xrange(y.shape[0]):
            rval[i, y[i]] = 1
        return rval

class MNISTDataProviderNoisy(MNISTDataProvider):
    """
    The class iterates over MNIST digits dataset, in possibly
    random order.
    """

    @staticmethod
    def apply_noise(img, noise_ratio=0):
        """
        add uniform noise to the image
        :img: 1D array of size 784 (MNIST Image)
        :noise_raito: the number of randomly selected cells where noise added between 0 and 784

        Usage/Doctest:
        >>> img = numpy.arange(784)
        >>> img_ns = MNISTDataProviderNoisy.apply_noise(img,  0.1)
        >>> (img == img_ns).all() #although there is extremely low chance that they will match
        False
        >>> img.shape == img_ns.shape
        True
        """
        noise_amnt = int(784 * noise_ratio)
        img_ns = numpy.copy(img)
        noise = numpy.random.rand(noise_amnt)
        #random pixels
        rnd_px = numpy.random.permutation(784)
        rnd_px = rnd_px[0:noise_amnt]
        img_ns[rnd_px] = noise
        return img_ns

    @staticmethod
    def apply_rotation(img, max_rot=0.0):
        from scipy import ndimage
        img_ns = img.reshape(28, 28)
        #between -max_rot, +max_rot
        rnd_ang = 2*max_rot * numpy.random.random_sample() - max_rot
        img_rt = ndimage.interpolation.rotate(img_ns, angle=rnd_ang, reshape=False)
        return img_rt.reshape(28*28)

    @staticmethod
    def apply_shift(img, max_shift=0.0):
        from scipy import ndimage
        img_ns = img.reshape(28, 28)
        #between -max_shift, +max_shift
        rnd_x = 2*max_shift * numpy.random.random_sample() - max_shift
        rnd_y = 2*max_shift * numpy.random.random_sample() - max_shift
        img_sht = ndimage.interpolation.shift(img_ns, shift=(rnd_x, rnd_y))
        return img_sht.reshape(28*28)


    def __init__(self, dset,
                 num_of_gen_data = 0,
                 noise_param = 0,
                 noise_fun = apply_noise,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 rng=None,
                 conv_reshape=False):

        self.name = "Noisy data provider"

        assert dset == "train", (
            "Noise is applied only to the training data"
        )

        super(MNISTDataProviderNoisy, self).\
            __init__(dset,
                     batch_size=batch_size,
                     max_num_batches=max_num_batches,
                     max_num_examples=max_num_examples,
                     randomize=True,
                     rng=rng,
                     conv_reshape=conv_reshape)

        assert num_of_gen_data >= 0, (
            "The number of generated data must be equal or above 0"
        )

        #selecting random images from original data
        rnd_ids = numpy.random.randint(low=0, high=self.x.shape[0], size=num_of_gen_data)

        noisy_x = numpy.zeros((num_of_gen_data, self.x.shape[1]))#numpy.copy(self.x[rnd_ids])
        noisy_t = numpy.copy(self.t[rnd_ids])


        for ns_id, rnd_id in enumerate(rnd_ids):
            noisy_x[ns_id] = noise_fun(self.x[rnd_id], noise_param)

        #mixing original and noisy data
        self.x = numpy.concatenate((self.x, noisy_x), axis=0)
        self.t = numpy.concatenate((self.t, noisy_t), axis=0)

        #mix the data, reset rand_idx
        self._rand_idx = None
        self.reset()


class MetOfficeDataProvider(DataProvider):
    """
    The class iterates over South Scotland Weather, in possibly
    random order.
    """
    def __init__(self, window_size,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True):

        super(MetOfficeDataProvider, self).\
            __init__(batch_size, randomize)

        dset_path = './data/HadSSP_daily_qc.txt'
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        if max_num_batches > 0 and max_num_examples > 0:
            logger.warning("You have specified both 'max_num_batches' and " \
                  "a deprecead 'max_num_examples' arguments. We will " \
                  "use the former over the latter.")
        
        raw = numpy.loadtxt(dset_path, skiprows=3, usecols=range(2, 32))
        
        self.window_size = window_size
        self._max_num_batches = max_num_batches
        #max_num_examples arg was provided for backward compatibility
        #but it maps us to the max_num_batches anyway
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size       
        
        #filter out all missing datapoints and
        #flatten a matrix to a vector, so we will get
        #a time preserving representation of measurments
        #with self.x[0] being the first day and self.x[-1] the last
        self.x = raw[raw >= 0].flatten()
        
        #normalise data to zero mean, unit variance
        mean = numpy.mean(self.x)
        var = numpy.var(self.x)
        assert var >= 0.01, (
            "Variance too small %f " % var
        )
        self.x = (self.x-mean)/var
        
        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MetOfficeDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        # we generate random indexes starting from window_size, i.e. 10th absolute element
        # in the self.x vector, as we later during mini-batch preparation slice
        # the self.x container backwards, i.e. given we want to get a training 
        # data-point for 11th day, we look at 10 preeceding days. 
        # Note, we cannot do this, for example, for the 5th day as
        # we do not have enough observations to make an input (10 days) to the model
        return numpy.random.permutation(numpy.arange(self.window_size, self.x.shape[0]))

    def next(self):

        has_enough = (self.window_size + self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches <= (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self.window_size + self._curr_idx, 
                             self.window_size + self._curr_idx + self.batch_size)

        #build slicing matrix of size minibatch, which will contain batch_size
        #rows, each keeping indexes that selects windows_size+1 [for (x,t)] elements
        #from data vector (self.x) that itself stays always sorted w.r.t time
        range_slices = numpy.zeros((self.batch_size, self.window_size + 1), dtype=numpy.int32)
       
        for i in xrange(0, self.batch_size):
            range_slices[i, :] = \
                numpy.arange(range_idx[i], 
                             range_idx[i] - self.window_size - 1, 
                             -1,
                             dtype=numpy.int32)[::-1]

        #here we use advanced indexing to select slices from observation vector
        #last column of rval_x makes our targets t (as we splice window_size + 1
        tmp_x = self.x[range_slices]
        rval_x = tmp_x[:,:-1]
        rval_t = tmp_x[:,-1].reshape(self.batch_size, -1)
        
        self._curr_idx += self.batch_size

        return rval_x, rval_t

    
class FuncDataProvider(DataProvider):
    """
    Function gets as an argument a list of functions defining the means
    of a normal distribution to sample from.
    """
    def __init__(self,
                 fn_list=[lambda x: x ** 2, lambda x: numpy.sin(x)],
                 std_list=[0.1, 0.1],
                 x_from = 0.0,
                 x_to = 1.0,
                 points_per_fn=200,
                 batch_size=10,
                 randomize=True):
        """
        """

        super(FuncDataProvider, self).__init__(batch_size, randomize)

        def sample_points(y, std):
            ys = numpy.zeros_like(y)
            for i in xrange(y.shape[0]):
                ys[i] = numpy.random.normal(y[i], std)
            return ys

        x = numpy.linspace(x_from, x_to, points_per_fn, dtype=numpy.float32)
        means = [fn(x) for fn in fn_list]
        y = [sample_points(mean, std) for mean, std in zip(means, std_list)]

        self.x_orig = x
        self.y_class = y

        self.x = numpy.concatenate([x for ys in y])
        self.y = numpy.concatenate([ys for ys in y])

        if self.randomize:
            self._rand_idx = self.__randomize()
        else:
            self._rand_idx = None

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        return numpy.random.permutation(numpy.arange(0, self.x.shape[0]))

    def __iter__(self):
        return self

    def next(self):
        if (self._curr_idx + self.batch_size) >= self.x.shape[0]:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        x = self.x[range_idx]
        y = self.y[range_idx]

        self._curr_idx += self.batch_size

        return x, y



