# %load mlp/maxpooling.py
import numpy as np
from mlp.conv import slide_tensor4
from mlp.layers import max_and_argmax, Layer
#from mlp.pool import pool_bc01, bprop_pool_bc01
import doctest

def pool_max(tensor4, window_shape, strides, prev_arg_max = None):
    """
    N - batch size
    C - num_in_chls
    W - image width
    H - image height
    S - window width
    R - window height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer
    u - stride x
    v - stride y
    :param tensor4:
    :param window_shape:
    :param strides:
    :return: (max_h of shape -> N x C x Q x P, indices of max values)

    Usage/Doctests
    non-overlapping example:
    >>> inputs = np.arange(16).reshape(1, 1, 4, 4)
    >>> inputs
    array([[[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11],
             [12, 13, 14, 15]]]])
    >>> window = (2, 2)
    >>> stride = (2, 2)
    >>> max_h, arg_max = pool_max(inputs, window, stride)
    >>> max_h
    array([[[[ 5,  7],
             [13, 15]]]])

    #testing arg_max by propagating our output max_h backwards
    >>> G_mat = np.zeros((inputs.shape[0],inputs.shape[1],max_h.shape[2]*max_h.shape[3], inputs.shape[2]*inputs.shape[3]))
    >>> G_mat[arg_max[0], arg_max[1], arg_max[2], arg_max[3]] = 1
    >>> G_mat = G_mat.swapaxes(3, 2)
    >>> rslt = np.einsum('ijkl,ijl->ijk', G_mat, max_h.reshape(inputs.shape[0], inputs.shape[1], -1))
    >>> rslt.reshape(inputs.shape)
    array([[[[  0.,   0.,   0.,   0.],
             [  0.,   5.,   0.,   7.],
             [  0.,   0.,   0.,   0.],
             [  0.,  13.,   0.,  15.]]]])

    Overlapping test case:
    >>> inputs = np.arange(9).reshape(1, 1, 3, 3)

    #this results in errors summation during backward propagation
    >>> inputs[0, 0, 1, 2] = 45
    >>> inputs
    array([[[[ 0,  1,  2],
             [ 3,  4, 45],
             [ 6,  7,  8]]]])
    >>> window = (2, 2)
    >>> stride = (1, 1) #results in overlapping windows
    >>> max_h, arg_max = pool_max(inputs, window, stride)
    >>> max_h
    array([[[[ 4, 45],
             [ 7, 45]]]])

    #testing arg_max by propagating our output max_h backwards
    >>> G_mat = np.zeros((inputs.shape[0],inputs.shape[1],max_h.shape[2]*max_h.shape[3], inputs.shape[2]*inputs.shape[3]))
    >>> G_mat[arg_max[0], arg_max[1], arg_max[2], arg_max[3]] = 1
    >>> G_mat = G_mat.swapaxes(3, 2)
    >>> t = np.einsum('ijkl,ijl->ijk', G_mat, max_h.reshape(inputs.shape[0], inputs.shape[1], -1))
    >>> t.reshape(inputs.shape)
    array([[[[  0.,   0.,   0.],
             [  0.,   4.,  90.],
             [  0.,   7.,   0.]]]])
    """

    t4 = tensor4 #N x C x W x H

    N, C, W, H = t4.shape
    S, R = window_shape
    u, v = strides

    Q = ((W - S) / u) + 1
    if (W - S) % u != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel width {1}, image width {2}".\
                         format(u, S, W))
    P = ((H - R) / v) + 1
    if (H - R) % v != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel height {1}, image height {2}".\
                         format(v, R, H))
    pam = prev_arg_max
    if pam is None:
        pam = np.zeros((4, N*C*Q*P), dtype=np.int64)
        pam[2] = np.tile(np.arange(Q*P), N*C)
        pam[1] = np.tile(np.repeat(np.arange(C), Q*P), N)
        pam[0] = np.repeat(np.arange(N), C*Q*P)

    t4_sld = slide_tensor4(t4, window_shape, strides)# N x C x Q*P x S*R
    max_h = np.amax(t4_sld, axis=3)# N x C x Q*P
    max_h = max_h.reshape(N, C, Q, P) # N x C x Q x P
    arg_max_new = np.argmax(t4_sld, axis=3) # N x C x Q*P
    arg_max_new = arg_max_new.reshape(-1)
    pam[3] = arg_max_new
    #y shift global: ((ma[2,:]%P)*v )
    #y shift inside:  ma[3,:]%R
    #x shift global: (ma[2,:]/P)*u
    #x shift inside: ma[3,:]/R
    pam[3:] =((pam[2,:]%P)*v + pam[3,:]%R) + ((pam[2,:]/P)*u+ pam[3,:]/R) * H
    return max_h, pam


class ConvMaxPool2D(Layer):
    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        self.chls = num_feat_maps
        self.wd = pool_shape #windoe
        self.stride = pool_stride
        self.inp_shp = None
        self.arg_max = None
        self.out_shp = None

    def fprop(self, inputs):
        if  self.inp_shp != inputs.shape:
            self.arg_max = None
            self.inp_shp = inputs.shape

        max_h, arg_max = pool_max(inputs, self.wd, self.stride, prev_arg_max=self.arg_max)
        self.arg_max = arg_max
        self.out_shp = max_h.shape
        return max_h

    def bprop(self, h, igrads, first_layer=False):
        #input from fully connected layer
        if first_layer:
            return None, None

        if self.out_shp is None:
            self.out_shp = h.shape
        #if igrads.ndim == 2:
        #    igrads = igrads.reshape(self.out_shp)

        inp_shp = self.inp_shp
        out_shp = self.out_shp
        am = self.arg_max
        G_mat = np.zeros((inp_shp[0], inp_shp[1], inp_shp[2]*inp_shp[3], out_shp[2]*out_shp[3]))
        G_mat[am[0], am[1], am[3], am[2]] = 1

        igrads = igrads.reshape(inp_shp[0], inp_shp[1], -1)

        #slow version
        #ograds = np.zeros((inp_shp[0], inp_shp[1], inp_shp[2]*inp_shp[3]))
        #for n in xrange(inp_shp[0]):
        #    for c in xrange(inp_shp[1]):
        #        ograds[n,c,:] = np.dot(G_mat[n,c,:,:], igrads[n,c,:])

        #crazy Einstein summation convention again!
        ograds = np.einsum('ijkl,ijl->ijk', G_mat, igrads)
        return None, ograds.reshape(inp_shp)

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvMaxPool2D.bprop_cost method not implemented')

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'

class ConvMaxPool2D_Cython(Layer):
    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D_Cython, self).__init__(rng=None)
        self.chls = num_feat_maps
        self.wnd = pool_shape #windoe
        self.stride = pool_stride
        self.inp_shp = None
        self.out_shp = None
        self.conv_shape = conv_shape

    def fprop(self, inputs):
        if  self.inp_shp != inputs.shape:
            self.inp_shp = inputs.shape

        num_imgs, num_chls = self.inp_shp[0], self.inp_shp[1]
        img_w, img_h = self.inp_shp[2], self.inp_shp[3]
        wnd_w, wnd_h = self.wnd
        stride_x, stride_y = self.stride
        out_w = (img_w - wnd_w) / stride_x + 1
        out_h = (img_h - wnd_h) / stride_y + 1
        inputs = inputs.astype(np.float32)
        
        self.memo = np.empty((num_imgs, num_chls) + (out_w, out_h) + (2,), dtype=np.int)
        imgs_out = np.empty((num_imgs, num_chls) + (out_w, out_h)).astype(np.float32)
        #print self.wd
        #print img_w, img_h
        #print out_w, out_h
        #print self.memo.shape, poolout.shape
        pool_max_cython(inputs, imgs_out, self.memo, wnd_w, wnd_h, stride_x, stride_y)
        return imgs_out

    def bprop(self, h, igrads, first_layer=False):
        #input from fully connected layer
        if first_layer:
            return None, None
        if self.out_shp is None:
            self.out_shp = h.shape
        if igrads.ndim == 2:
            igrads = igrads.reshape(self.out_shp)
        igrads = igrads.astype(np.float32)
        ograds = np.empty((self.inp_shp), dtype=np.float32)
        pool_max_cython_bp(igrads, self.memo, ograds)
        return None, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvMaxPool2D.bprop_cost method not implemented')

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d_cython'


if __name__ == "__main__":
    #tests via doctest https://docs.python.org/2/library/doctest.html#module-doctest
    #to see detailed ouptut run as:
    #python ConvMaxPool.py -v
    doctest.testmod()