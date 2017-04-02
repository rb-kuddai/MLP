from __future__ import division
import numpy as np
import cython
cimport numpy as np


ctypedef Py_ssize_t uint
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def pool_max_cython(np.ndarray[DTYPE_t, ndim=4] imgs,
              np.ndarray[DTYPE_t, ndim=4] imgs_out,
              np.ndarray[np.int_t, ndim=5] memo,
              uint wnd_w, uint wnd_h, uint stride_x, uint stride_y):
    """
    Max Pooling
    Based on the https://github.com/andersbll/nnet/blob/master/nnet/convnet/conv.pyx
    Note that my implementation accounts custom striding. I count pool window completely
    differently
    imgs has shape (num_imgs, num_chls, img_w, img_h)
    imgs_out has shape (num_imgs, num_chls, (img_w - pool_w) // stride_x + 1 , (img_h - pool_h) // stride_y + 1)
    memo has shape (num_imgs, n_chls, (img_w - pool_w) // stride_x + 1, (img_h - pool_h) // stride_y + 1,  2)
    """

    cdef uint num_imgs = imgs.shape[0]
    cdef uint num_chls = imgs.shape[1]
    cdef uint img_w = imgs.shape[2]
    cdef uint img_h = imgs.shape[3]

    cdef uint out_w = (img_w - wnd_w) // stride_x + 1
    cdef uint out_h = (img_h - wnd_h) // stride_y + 1

    if (img_w - wnd_w) % stride_x != 0:
        raise ValueError('Impossible to tile along width.')
    if (img_h - wnd_h) % stride_y != 0:
        raise ValueError('Impossible to tile along height.')
    if not num_imgs == imgs_out.shape[0] == memo.shape[0]:
        raise ValueError('Number of images is not correct. {} {} {}'.format(num_imgs, imgs_out.shape[0], memo.shape[0]))
    if not num_chls == imgs_out.shape[1] == memo.shape[1]:
        raise ValueError('Number of channels is not correct. {} {} {}'.format(num_chls, imgs_out.shape[1], memo.shape[1]))
    if not (out_h == imgs_out.shape[2] == memo.shape[2] and out_w == imgs_out.shape[3] == memo.shape[3]):
        raise ValueError('Image shape is not correct. {}, {}, {}'.format(out_h, imgs_out.shape[2], memo.shape[2]))
    if not memo.shape[4] == 2:
        raise ValueError('Memo last axe is not correct. Must store max coordinates')

    #inp - input, wnd - window
    cdef uint i, c, y_inp, x_inp, y_out, x_out, wnd_x, wnd_y, pool_x, pool_y
    cdef uint img_y_max = 0
    cdef uint img_x_max = 0
    cdef DTYPE_t max_value = 0.0
    cdef DTYPE_t new_value =0.0

    for i in range(num_imgs):
        for c in range(num_chls):
            for x_out in range(out_w):
                wnd_x = x_out * stride_x
                for y_out in range(out_h):
                    wnd_y = y_out * stride_y
                    max_value = -1e30
                    for pool_x in range(wnd_w):
                        x_inp = wnd_x + pool_x
                        for pool_y in range(wnd_h):
                            y_inp = wnd_y + pool_y
                            new_value = imgs[i, c, x_inp, y_inp]
                            if new_value > max_value:
                                max_value = new_value
                                img_x_max = x_inp
                                img_y_max = y_inp
                    imgs_out[i, c, x_out, y_out] = max_value
                    memo[i, c, x_out, y_out, 0] = img_x_max
                    memo[i, c, x_out, y_out, 1] = img_y_max

#bp - back propagating

@cython.boundscheck(False)
@cython.wraparound(False)
def pool_max_cython_bp(np.ndarray[DTYPE_t, ndim=4] igrads,
                    np.ndarray[np.int_t, ndim=5] memo,
                    np.ndarray[DTYPE_t, ndim=4] ograds):

    cdef uint num_imgs = igrads.shape[0]
    cdef uint num_chls = igrads.shape[1]
    cdef uint igrads_w = igrads.shape[2]
    cdef uint igrads_h = igrads.shape[3]

    cdef uint i, c, y, x, img_y, img_x

    ograds[...] = 0
    for i in range(num_imgs):
        for c in range(num_chls):
            for x in range(igrads_w):
                for y in range(igrads_h):
                    img_x = memo[i, c, x, y, 0]
                    img_y = memo[i, c, x, y, 1]
                    #here was the bug in the orignal code
                    ograds[i, c, img_x, img_y] += igrads[i, c, x, y]