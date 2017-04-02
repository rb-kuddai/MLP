
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import doctest
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import logging
from mlp.layers import Layer

logger = logging.getLogger(__name__)

"""
conv.py:
Most of the code in this section is doc tests. This is tests which are implemented in
comments/function_description and they are started with >>>.
More about this: https://docs.python.org/2/library/doctest.html
I need so many tests because I have implemented convolution via numpy matrix dot products
and numpy strides manipulations. That is why this implementation is quite fast,
but I had a lot of troubles with indices.

The test in the description for my_conv_fwd function is taken from this paper:
Chellapilla, Kumar, Sidd Puri, and Patrice Simard.
"High performance convolutional neural networks for document processing."
Tenth International Workshop on Frontiers in Handwriting Recognition. Suvisoft, 2006

IMPORTANT!!! This is implementation works with non-standard strides as well.
In order to achieve this during back propagation the igrad has to be padded not only outside
by (kernel_size - 1) but inside by (stride - 1) as well.
In case of stride (2, 2) and kernel (3,3) igrad will be padded before convolution like so:
             0 0 0 0 0 0 0
             0 0 0 0 0 0 0
  1 2  ->    0 0 1 0 2 0 0
  3 4  ->    0 0 0 0 0 0 0
             0 0 3 0 4 0 0
             0 0 0 0 0 0 0
             0 0 0 0 0 0 0

I put everything in one file because it has been said to do so in the task descripton
"""


def slide_tensor4(tensor4, window_shape, strides):
    """
    :param tensor4:  minibatch size x number of channels x image height x image width
    :param window_shape: (window_height, window_width)
    :param strides: (height_step, width_step)
    :return: tensor4 -> minibatch size x number of channels x number of windows in image x window_width * window_height

    Description:
    Slide tensor4 by window with strides step. For that it just
    changes tensor4 internal numpy stride representation so it must be very fast.
    Between window and image shape the following must be true:
    ((img_shp - window_shape) / strides) + 1 == integer
    Otherwise it is impossible to tile images equally.

    Usage/Doctests:
    Simple example 2x2 window with step 1 along each dimension.
    In 3x3 image 4 windows in total
    >>> tensor4 = np.arange(9).reshape(1, 1, 3, 3)
    >>> tensor4
    array([[[[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]]])
    >>> slide_tensor4(tensor4, (2,2), (1,1))
    array([[[[0, 1, 3, 4],
             [1, 2, 4, 5],
             [3, 4, 6, 7],
             [4, 5, 7, 8]]]])

    >>> tensor4 = np.arange(9).reshape(1, 1, 3, 3)
    >>> tensor4
    array([[[[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]]])
    >>> slide_tensor4(tensor4, (2,3), (1,1))
    array([[[[0, 1, 2, 3, 4, 5],
             [3, 4, 5, 6, 7, 8]]]])

    >>> tensor4 = np.arange(12).reshape(1, 1, 4, 3)
    >>> tensor4
    array([[[[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8],
             [ 9, 10, 11]]]])
    >>> slide_tensor4(tensor4, (2,2), (2,1))
    array([[[[ 0,  1,  3,  4],
             [ 1,  2,  4,  5],
             [ 6,  7,  9, 10],
             [ 7,  8, 10, 11]]]])

    Example with windows without overlapping. 4 windows in total
    >>> tensor4 = np.arange(36).reshape(1, 1, 6, 6)
    >>> tensor4
    array([[[[ 0,  1,  2,  3,  4,  5],
             [ 6,  7,  8,  9, 10, 11],
             [12, 13, 14, 15, 16, 17],
             [18, 19, 20, 21, 22, 23],
             [24, 25, 26, 27, 28, 29],
             [30, 31, 32, 33, 34, 35]]]])
    >>> slide_tensor4(tensor4, (3,3), (3,3))
    array([[[[ 0,  1,  2,  6,  7,  8, 12, 13, 14],
             [ 3,  4,  5,  9, 10, 11, 15, 16, 17],
             [18, 19, 20, 24, 25, 26, 30, 31, 32],
             [21, 22, 23, 27, 28, 29, 33, 34, 35]]]])

    #Example with 2 images in minibatch and 2 channels
    >>> tensor4 = np.arange(36).reshape(2, 2, 3, 3)
    >>> tensor4
    array([[[[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8]],
    <BLANKLINE>
            [[ 9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[18, 19, 20],
             [21, 22, 23],
             [24, 25, 26]],
    <BLANKLINE>
            [[27, 28, 29],
             [30, 31, 32],
             [33, 34, 35]]]])
    >>> slide_tensor4(tensor4, (2,2), (1,1))
    array([[[[ 0,  1,  3,  4],
             [ 1,  2,  4,  5],
             [ 3,  4,  6,  7],
             [ 4,  5,  7,  8]],
    <BLANKLINE>
            [[ 9, 10, 12, 13],
             [10, 11, 13, 14],
             [12, 13, 15, 16],
             [13, 14, 16, 17]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[18, 19, 21, 22],
             [19, 20, 22, 23],
             [21, 22, 24, 25],
             [22, 23, 25, 26]],
    <BLANKLINE>
            [[27, 28, 30, 31],
             [28, 29, 31, 32],
             [30, 31, 33, 34],
             [31, 32, 34, 35]]]])
    """
    t4 = tensor4
    #convert to array for easy algebraic manipulations
    wd = np.array(window_shape)
    ss = np.array(strides)
    img_shp = np.array(t4.shape[2:4])

    dim_check = ((img_shp - wd) % ss)
    if dim_check[0] != 0 or dim_check[1] != 0:
        raise ValueError("impossible to tile with given strides {0}, window shape {1}, tensor4 shape {2}"\
                        .format(strides, window_shape, tensor4.shape))

    slide_shp = ((img_shp - wd) / ss) + 1
    #using tuple addition (1, 2) + (3, 4) = (1, 2, 3, 4)
    new_shp = t4.shape[0:2] + tuple(slide_shp) + tuple(wd)
    new_str = t4.strides[0:2] + tuple(np.array(t4.strides[2:4]) * ss) + t4.strides[2:4]
    #change internal memory numpy array t4 representation
    #more here: http://www.johnvinyard.com/blog/?p=268
    strided = ast(t4, shape = new_shp, strides = new_str)
    final_shape = t4.shape[0:2] +(slide_shp.prod(),) +(wd.prod(),)
    return strided.reshape(final_shape)

def slide_tensor4_2(tensor4, window_shape, strides):
    t4 = tensor4
    #convert to array for easy algebraic manipulations
    wd = np.array(window_shape)
    ss = np.array(strides)
    img_shp = np.array(t4.shape[2:4])

    dim_check = ((img_shp - wd) % ss)
    if dim_check[0] != 0 or dim_check[1] != 0:
        raise ValueError("impossible to tile with given strides {0}, window shape {1}, tensor4 shape {2}"\
                        .format(strides, window_shape, tensor4.shape))

    slide_shp = ((img_shp - wd) / ss) + 1
    #using tuple addition (1, 2) + (3, 4) = (1, 2, 3, 4)
    new_shp = t4.shape[0:2] + tuple(slide_shp) + tuple(wd)
    new_str = t4.strides[0:2] + tuple(np.array(t4.strides[2:4]) * ss) + t4.strides[2:4]
    #change internal memory numpy array t4 representation
    #more here: http://www.johnvinyard.com/blog/?p=268
    return ast(t4, shape = new_shp, strides = new_str)

def my_conv_fwd(image, kernels, strides=(1, 1)):
    """
    Description:
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    N - batch size
    C - num_in_chls
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer

    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param kernels: 4D tensor of kernels of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)

    Usage/Doctest:
    The test is taken from the paper:
    Chellapilla, Kumar, Sidd Puri, and Patrice Simard.
    "High performance convolutional neural networks for document processing."
    Tenth International Workshop on Frontiers in Handwriting Recognition. Suvisoft, 2006.

    >>> image = np.empty((1, 3, 3, 3), dtype=np.int64)
    >>> image[0, 0, :, :] = np.array([[1, 2, 0], [1, 1, 3], [0, 2, 2]])
    >>> image[0, 1, :, :] = np.array([[0, 2, 1], [0, 3, 2], [1, 1, 0]])
    >>> image[0, 2, :, :] = np.array([[1, 2, 1], [0, 1, 3], [3, 3, 2]])

    >>> kernel = np.empty((3, 2, 2, 2), dtype=np.int64)
    >>> kernel[0, 0, :, :] = np.array([[1, 1], [2, 2]])
    >>> kernel[1, 0, :, :] = np.array([[1, 1], [1, 1]])
    >>> kernel[2, 0, :, :] = np.array([[0, 1], [1, 0]])

    >>> kernel[0, 1, :, :] = np.array([[1, 0], [0, 1]])
    >>> kernel[1, 1, :, :] = np.array([[2, 1], [2, 1]])
    >>> kernel[2, 1, :, :] = np.array([[1, 2], [2, 0]])

    >>> img = image.swapaxes(2, 3)
    >>> krl = kernel.swapaxes(2, 3)
    >>> my_conv_fwd(img, krl)
    array([[[[14, 15],
             [20, 24]],
    <BLANKLINE>
            [[12, 17],
             [24, 26]]]])
    """
    #swaps X, Y axis due to the differences in conventions with slide_tensor4
    img = image.swapaxes(2, 3)  #N x C x H x W
    krl = kernels.swapaxes(2, 3)#C x K x R x S
    strd = strides[::-1]

    N, C, H, W = img.shape
    C, K, R, S = krl.shape

    Q = ((W - S) / strd[0]) + 1
    if (W - S) % strd[0] != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel width {1}, image width {2}".\
                         format(strd[0], S, W))
    P = ((H - R) / strd[1]) + 1
    if (H - R) % strd[1] != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel height {1}, image height {2}".\
                         format(strd[1], R, H))

    krl = krl.swapaxes(0, 1) #K x C x R x S
    krl = krl.reshape(K, C*R*S) #K x C*R*S
    krl = krl.swapaxes(0, 1) #C*R*S x K

    sld_img = slide_tensor4(img, (R,S), strd) # N x C x P*Q x R*S
    sld_img = sld_img.swapaxes(1, 2) # N x P*Q x C x R*S
    sld_img = sld_img.reshape(N*P*Q, C*R*S) # N*P*Q x C*R*S

    rslt = np.dot(sld_img, krl) # dot(N*P*Q x C*R*S, C*R*S x K) = N*P*Q x K
    rslt = rslt.swapaxes(0, 1) # K x N*P*Q
    rslt = rslt.reshape((K, N, P, Q)) # K x N x P x Q
    rslt = rslt.swapaxes(0, 1) # N x K x P x Q
    rslt = rslt.swapaxes(2, 3) # N x K x Q x P
    return rslt


def my_conv_fwd2(image, kernels, strides=(1, 1)):
    """
    Description:
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    N - batch size
    C - num_in_chls
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer

    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param kernels: 4D tensor of kernels of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)

    Usage/Doctest:
    The test is taken from the paper:
    Chellapilla, Kumar, Sidd Puri, and Patrice Simard.
    "High performance convolutional neural networks for document processing."
    Tenth International Workshop on Frontiers in Handwriting Recognition. Suvisoft, 2006.

    >>> image = np.empty((1, 3, 3, 3), dtype=np.int64)
    >>> image[0, 0, :, :] = np.array([[1, 2, 0], [1, 1, 3], [0, 2, 2]])
    >>> image[0, 1, :, :] = np.array([[0, 2, 1], [0, 3, 2], [1, 1, 0]])
    >>> image[0, 2, :, :] = np.array([[1, 2, 1], [0, 1, 3], [3, 3, 2]])

    >>> kernel = np.empty((3, 2, 2, 2), dtype=np.int64)
    >>> kernel[0, 0, :, :] = np.array([[1, 1], [2, 2]])
    >>> kernel[1, 0, :, :] = np.array([[1, 1], [1, 1]])
    >>> kernel[2, 0, :, :] = np.array([[0, 1], [1, 0]])

    >>> kernel[0, 1, :, :] = np.array([[1, 0], [0, 1]])
    >>> kernel[1, 1, :, :] = np.array([[2, 1], [2, 1]])
    >>> kernel[2, 1, :, :] = np.array([[1, 2], [2, 0]])

    >>> img = image.swapaxes(2, 3)
    >>> krl = kernel.swapaxes(2, 3)
    >>> my_conv_fwd2(img, krl)
    array([[[[14, 15],
             [20, 24]],
    <BLANKLINE>
            [[12, 17],
             [24, 26]]]])
    """
    #swaps X, Y axis due to the differences in conventions with slide_tensor4
    img = image#N x C x W x H #image.swapaxes(2, 3)  #N x C x H x W
    krl = kernels#C x K x S x R #kernels.swapaxes(2, 3)#C x K x R x S
    strd = strides# strides[::-1]

    N, C, W, H = img.shape
    C, K, S, R = krl.shape

    Q = ((W - S) / strd[0]) + 1
    if (W - S) % strd[0] != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel width {1}, image width {2}".\
                         format(strd[0], S, W))
    P = ((H - R) / strd[1]) + 1
    if (H - R) % strd[1] != 0:
        raise ValueError("impossible to tile with given stride {0}, kernel height {1}, image height {2}".\
                         format(strd[1], R, H))

    #krl#C x K x S x R
    sld_img = slide_tensor4_2(img, (S,R), strd) # N x C x Q x P x S x R
    rslt = np.tensordot(sld_img, krl, axes=([1,4,5],[0,2,3])) #N x Q x P x K
    rslt = np.rollaxis(rslt, 3, 1) #N x K x Q x P
    return rslt

def my_conv_bck(igrad, kernels, strides=(1, 1)):
    """
    current layer:
    N - batch size
    C - num_in_chls #number of input channels
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer
    u - stride x
    v - stride y
    :param igrad: 4D tensor of sizes (N, K, Q, P)
    :param kernels: 4D tensor of  kernels of size (C, K, S, R)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: delta 4D tensor of size (N, C, W, H)

    Usage/Doctest:
    The case with kernel shape = (3,3) and strides = (2,2)
    input to the layer during forward propagation has shape = (1, 1, 5, 5)
    output has shape (1, 1, 2, 2) accordingly and igrad has the same shape
    so igrad:
    1 2
    3 4

    should be turned into:
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 1 0 2 0 0
    0 0 0 0 0 0 0
    0 0 3 0 4 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0

    for proper convolution and after that it must convolved with rotated kernels.
    The padding on the edge is to account kernel size.
    Note! The padding in between to account strides
    >>> inputs = np.arange(25).reshape(1, 1, 5, 5)
    >>> kernel = np.arange(9).reshape(1, 1, 3, 3)
    >>> igrad = np.arange(4).reshape(1, 1, 2, 2)
    >>> ograd = my_conv_bck(igrad, kernel, strides=(2, 2))
    >>> ograd.shape == inputs.shape
    True

    The case with different number of input feature maps and output feature maps
    >>> inputs = np.arange(54).reshape(3, 2, 3, 3)
    >>> kernel = np.arange(24).reshape(2, 3, 2, 2)
    >>> igrad = np.arange(36).reshape(3, 3, 2, 2)
    >>> ograd = my_conv_bck(igrad, kernel, strides=(1, 1))
    >>> ograd.shape == inputs.shape
    True
    """
    N, K, Q, P = igrad.shape
    C, K, S, R = kernels.shape
    u, v = strides
    img, krl = igrad, kernels
    #IMPORTANT! To account strides
    #In case of kernel shape (3,3), strides(2,2)
    if u > 1 or v > 1:
        tmp = np.zeros( (N, K, Q + (u - 1)*(Q - 1), P + (v - 1)*(P - 1)) )
        tmp[:,:,::u,::v]=img
        img = tmp

    img = pad_tensor4(img, (S - 1, R - 1))# (N, K, Q + (u - 1)*(Q-1) + 2*(S - 1), P + (v - 1)*(P - 1) + 2*(R - 1))
    krl = tensor4_rot180(krl)# (C, K, rot180(S, R))
    #swap here because we are going backwards
    krl = krl.swapaxes(0, 1)# (K, C, rot180(S, R))
    #strides = (1,1) not a typo!
    #The strides(u,v) have been accounted above already by tiling img with zeros in between
    return my_conv_fwd(img, krl, strides=(1,1)) #(N, C, W, H)

def my_conv_bck2(igrad, kernels, strides=(1, 1)):
    """
    current layer:
    N - batch size
    C - num_in_chls #number of input channels
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer
    u - stride x
    v - stride y
    :param igrad: 4D tensor of sizes (N, K, Q, P)
    :param kernels: 4D tensor of  kernels of size (C, K, S, R)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: delta 4D tensor of size (N, C, W, H)

    Usage/Doctest:
    The case with kernel shape = (3,3) and strides = (2,2)
    input to the layer during forward propagation has shape = (1, 1, 5, 5)
    output has shape (1, 1, 2, 2) accordingly and igrad has the same shape
    so igrad:
    1 2
    3 4

    should be turned into:
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0
    0 0 1 0 2 0 0
    0 0 0 0 0 0 0
    0 0 3 0 4 0 0
    0 0 0 0 0 0 0
    0 0 0 0 0 0 0

    for proper convolution and after that it must convolved with rotated kernels.
    The padding on the edge is to account kernel size.
    Note! The padding in between to account strides
    >>> inputs = np.arange(25).reshape(1, 1, 5, 5)
    >>> kernel = np.arange(9).reshape(1, 1, 3, 3)
    >>> igrad = np.arange(4).reshape(1, 1, 2, 2)
    >>> ograd = my_conv_bck2(igrad, kernel, strides=(2, 2))
    >>> ograd.shape == inputs.shape
    True

    The case with different number of input feature maps and output feature maps
    >>> inputs = np.arange(54).reshape(3, 2, 3, 3)
    >>> kernel = np.arange(24).reshape(2, 3, 2, 2)
    >>> igrad = np.arange(36).reshape(3, 3, 2, 2)
    >>> ograd = my_conv_bck2(igrad, kernel, strides=(1, 1))
    >>> ograd.shape == inputs.shape
    True
    """
    N, K, Q, P = igrad.shape
    C, K, S, R = kernels.shape
    u, v = strides
    img, krl = igrad, kernels
    #IMPORTANT! To account strides
    #In case of kernel shape (3,3), strides(2,2)
    if u > 1 or v > 1:
        tmp = np.zeros( (N, K, Q + (u - 1)*(Q - 1), P + (v - 1)*(P - 1)) )
        tmp[:,:,::u,::v]=img
        img = tmp

    img = pad_tensor4(img, (S - 1, R - 1))# (N, K, Q + (u - 1)*(Q-1) + 2*(S - 1), P + (v - 1)*(P - 1) + 2*(R - 1))
    krl = tensor4_rot180(krl)# (C, K, rot180(S, R))
    #swap here because we are going backwards
    krl = krl.swapaxes(0, 1)# (K, C, rot180(S, R))
    #strides = (1,1) not a typo!
    #The strides(u,v) have been accounted above already by tiling img with zeros in between
    return my_conv_fwd2(img, krl, strides=(1,1)) #(N, C, W, H)

def my_conv_grad(inputs, deltas, strides=(1, 1)):
    """
    implementation references:
    http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
    http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/

    current layer:
    N - batch size
    C - num_in_chls #number of input channels
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer

    :inputs: tensor4 with shape -> (N, C, W, H)  #It has the same shape as input, or ograd
    :deltas: tensor4 with shape -> (N, K, Q, P)  #It has the same shape as output or igrad
    :return: tensor4 with shape -> (C, K, S, R)  #It has the same shape as kernel

    Usage/Doctest:
    Test case with non-standard strides = (2, 2)
    As we can see the derivatives of kernel has the same shape as kernel which is good
    >>> inputs = np.arange(50).reshape(1, 2, 5, 5)
    >>> kernel = np.arange(54).reshape(2, 3, 3, 3)
    >>> deltas = np.arange(12).reshape(1, 3, 2, 2)
    >>> kernel_grads = my_conv_grad(inputs, deltas, strides=(2, 2))
    >>> kernel_grads.shape == kernel.shape
    True
    """
    dlt = deltas
    #swaps X, Y axis due to the differences in conventions with slide_tensor4
    ipt = inputs.swapaxes(2, 3) #N x C x H x W
    dlt = dlt.swapaxes(2, 3) #N x K x P x Q
    strd = strides[::-1]

    N, C, H, W = ipt.shape
    N, K, P, Q = dlt.shape
    #inverse expression to get kernel width and height
    S = W - (Q - 1) * strd[0]
    R = H - (P - 1) * strd[1]

    dlt = dlt.swapaxes(0, 1) #K x N x P x Q
    dlt = dlt.reshape(K, N*P*Q) #K x N*P*Q

    sld_img = slide_tensor4(ipt, (R,S), strd) # N x C x P*Q x R*S
    sld_img = sld_img.swapaxes(1, 2) # N x P*Q x C x R*S
    sld_img = sld_img.reshape(N*P*Q, C*R*S) # N*P*Q x C*R*S

    rslt = np.dot(dlt, sld_img) #dot(K x N*P*Q, N*P*Q x C*R*S) = K x C*R*S
    rslt = rslt.reshape((K, C, R, S)) # K x C x R x S
    rslt = rslt.swapaxes(0, 1) # C x K x R x S
    rslt = rslt.swapaxes(2, 3) # C x K x S x R
    return rslt

def my_conv_grad2(inputs, deltas, strides=(1, 1)):
    """
    implementation references:
    http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
    http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/

    current layer:
    N - batch size
    C - num_in_chls #number of input channels
    W - image width
    H - image height
    K - num_out_chls
    S - kernel width
    R - kernel height
    Q - output width - ((W - S) / strides[0]) + 1 == integer
    P - output height - ((H - R) / strides[1]) + 1 == integer

    :inputs: tensor4 with shape -> (N, C, W, H)  #It has the same shape as input, or ograd
    :deltas: tensor4 with shape -> (N, K, Q, P)  #It has the same shape as output or igrad
    :return: tensor4 with shape -> (C, K, S, R)  #It has the same shape as kernel

    Usage/Doctest:
    Test case with non-standard strides = (2, 2)
    As we can see the derivatives of kernel has the same shape as kernel which is good
    >>> inputs = np.arange(50).reshape(1, 2, 5, 5)
    >>> kernel = np.arange(54).reshape(2, 3, 3, 3)
    >>> deltas = np.arange(12).reshape(1, 3, 2, 2)
    >>> kernel_grads = my_conv_grad2(inputs, deltas, strides=(2, 2))
    >>> kernel_grads.shape == kernel.shape
    True
    """
    dlt = deltas #N x K x Q x P
    ipt = inputs #N x C x W x H #inputs.swapaxes(2, 3) #N x C x H x W
     #dlt.swapaxes(2, 3) #N x K x P x Q
    strd = strides #strides[::-1]

    N, C, W, H = ipt.shape
    N, K, Q, P = dlt.shape
    #inverse expression to get kernel width and height
    S = W - (Q - 1) * strd[0]
    R = H - (P - 1) * strd[1]
                                                 #N x K x Q x P
    sld_img = slide_tensor4_2(ipt, (S,R), strd) # N x C x Q x P x S x R
    rslt = np.tensordot(sld_img, dlt, axes=([0,2,3],[0,2,3])) #C x S x R x K
    rslt = np.rollaxis(rslt, 3, 1) #C x K x S x R
    return rslt # C x K x S x R

def tile_bias(output_shape, bias):
    """
    :output_shape: shape of (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    :bias: tensor1D of shape (num_out_feature_maps)
    :return: bias of shape (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)

    Usage/Doctest:
    >>> b = np.array([1, 3])
    >>> output_shape = (1, 2, 3, 3)
    >>> rslt = tile_bias(output_shape, b)
    >>> rslt[:, 0, :, :]
    array([[[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]])
    >>> rslt[:, 1, :, :]
    array([[[3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]]])
    """
    return ast(bias, shape=output_shape, strides=(0, bias.strides[0], 0, 0))

def tensor4_rot180(tensor4):
    """
    C - num_in_chls #number of input channels
    K - num_out_chls
    S - kernel width
    R - kernel height

    :param tensor4: shape (C, K, S, R)
    :return: (C, K, rot180(S, R))

    Usage/Doctest:
    >>> ar = np.arange(9).reshape(1, 1, 3, 3)
    >>> ar
    array([[[[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]]])
    >>> tensor4_rot180(ar)
    array([[[[8, 7, 6],
             [5, 4, 3],
             [2, 1, 0]]]])

    >>> ar = np.arange(18).reshape(1, 2, 3, 3)
    >>> ar
    array([[[[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8]],
    <BLANKLINE>
            [[ 9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]]]])
    >>> tensor4_rot180(ar)
    array([[[[ 8,  7,  6],
             [ 5,  4,  3],
             [ 2,  1,  0]],
    <BLANKLINE>
            [[17, 16, 15],
             [14, 13, 12],
             [11, 10,  9]]]])
    """
    t = tensor4.swapaxes(0, 2)#(S, K, C, R)
    t = t.swapaxes(1, 3)#(S, R, C, K)
    t = np.rot90(t, 2) #(rot180(S, R), C, K)
    t = t.swapaxes(0, 2)#(C, K, S, R) still rotated by 180
    t = t.swapaxes(1, 3)#(C, R, rot180(S, K))
    return t

def pad_tensor4(tensor4, pad_shape):
    """
    :param tensor4: tensor4 of shape: (batch size, num channels, width, height)
    :param pad_shape: (pad_width, pad_height)
    :return: tensor4 of shape (batch size, num channels, width + pad_width*2, height + pad_height*2)

    Usage/Doctest:
    >>> ar = np.arange(9).reshape(1, 1, 3, 3)
    >>> pad_tensor4(ar, (1,2))
    array([[[[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 2, 0, 0],
             [0, 0, 3, 4, 5, 0, 0],
             [0, 0, 6, 7, 8, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]]])
    """
    w, h = pad_shape
    return np.lib.pad(tensor4, ((0,0), (0,0), (w,w), (h,h)), 'constant', constant_values=0)

class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my_conv_fwd,
                 conv_bck=my_conv_bck,
                 conv_grad=my_conv_grad):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """
        super(ConvLinear, self).__init__(rng=rng)

        #chls - channels
        self.inp_chls = num_inp_feat_maps
        self.out_chls = num_out_feat_maps
        self.img_shp = image_shape
        self.krl_shp = kernel_shape #krl - kernel
        self.stride = stride

        self.conv_fwd = conv_fwd
        self.conv_bck = conv_bck
        self.conv_grad = conv_grad

        self.W = self.rng.uniform(
            -irange, irange,
            (self.inp_chls, self.out_chls) + self.krl_shp)
        self.b = np.zeros((self.out_chls,), dtype=np.float32)

    def fprop(self, inputs):
        """
        kernel -> (num_in_feature_maps, num_out_feature_maps, kernel_width, kernel_height)
        feature_map_width -> (image_width - kernel_width) / strides[0] + 1
        :param inputs: (batch_size, num_in_feature_maps, image_width, image_height)
        :return: output tensor4 of size: (batch_size, num_out_feature_maps, feature_map_width, feature_map_height)
        """
        #output shape: (batch_size, num_out_feature_maps, feature_map_width, feature_map_height)
        output = self.conv_fwd(inputs, self.W, self.stride)
        return output + tile_bias(output.shape, self.b)

    def bprop(self, h, igrads, first_layer=False):
        """
        kernel -> (num_in_feature_maps, num_out_feature_maps, kernel_width, kernel_height)
        feature_map_width -> (image_width - kernel_width) / strides[0] + 1
        :param h: tensor4 of size: (batch_size, num_out_feature_maps, feature_map_width, feature_map_height)
        :param igrads: tensor4 of size: (batch_size, num_out_feature_maps, feature_map_width, feature_map_height)
        :return:
                deltas: tensor4 with shape (batch_size, num_out_feature_maps, feature_map_width, feature_map_height)
                ograds: tensor4 with shape (batch_size, num_in_feature_maps, image_width, image_width)
        """
        #input from fully connected layer

        if igrads.ndim == 2:
            igrads = igrads.reshape(h.shape)
        if first_layer:
            return  igrads, None
        ograds = self.conv_bck(igrads, self.W, self.stride)
        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        """
        implementation references:
        http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
        http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/

        current layer:
        N - batch size
        C - num_in_chls #number of input channels
        W - image width
        H - image height
        K - num_out_chls
        S - kernel width
        R - kernel height
        Q - output width - ((W - S) / strides[0]) + 1 == integer
        P - output height - ((H - R) / strides[1]) + 1 == integer

        :inputs: tensor4 with shape (N, C, W, H)
        :deltas: tensor4 with shape (N, K, Q, P)
        """
        #same code as in layers.py to account regularization
        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight*self.W
            l2_b_penalty = l2_weight*self.b

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight*np.sign(self.W)
            l1_b_penalty = l1_weight*np.sign(self.b)

        W_grads = self.conv_grad(inputs, deltas, self.stride) + l1_W_penalty + l2_W_penalty
        b_grads = np.sum(deltas, axis=(0, 2, 3)) + l1_b_penalty + l2_b_penalty
        return  [W_grads, b_grads]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'


class ConvSigmoid(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my_conv_fwd,
                 conv_bck=my_conv_bck,
                 conv_grad=my_conv_grad):

        super(ConvSigmoid, self).__init__(num_inp_feat_maps = num_inp_feat_maps,
                                         num_out_feat_maps = num_out_feat_maps,
                                         image_shape = image_shape,
                                         kernel_shape = kernel_shape,
                                         stride = stride,
                                         irange = irange,
                                         rng = rng,
                                         conv_fwd = conv_fwd,
                                         conv_bck = conv_bck,
                                         conv_grad = conv_grad)

    def fprop(self, inputs):
        #identical to Sigmoid implementation in layers.py
        a = super(ConvSigmoid, self).fprop(inputs)
        np.clip(a, -30.0, 30.0, out=a)
        h = 1.0/(1.0 + np.exp(-a))
        return h

    def bprop(self, h, igrads, first_layer=False):
        #identical to Sigmoid implementation in layers.py
        #input from fully connected layer
        if igrads.ndim == 2:
            igrads = igrads.reshape(h.shape)
        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(ConvSigmoid, self).bprop(h=None, igrads=deltas, first_layer=first_layer)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvSigmoid.bprop_cost method not implemented')

    def get_name(self):
        return 'convsigmoid'


class ConvRelu(ConvLinear):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my_conv_fwd,
                 conv_bck=my_conv_bck,
                 conv_grad=my_conv_grad):

        super(ConvRelu, self).__init__(num_inp_feat_maps = num_inp_feat_maps,
                                         num_out_feat_maps = num_out_feat_maps,
                                         image_shape = image_shape,
                                         kernel_shape = kernel_shape,
                                         stride = stride,
                                         irange = irange,
                                         rng = rng,
                                         conv_fwd = conv_fwd,
                                         conv_bck = conv_bck,
                                         conv_grad = conv_grad)

    def fprop(self, inputs):
        #identical to Relu implementation in layers.py
        a = super(ConvRelu, self).fprop(inputs)
        h = np.clip(a, 0, 20.0)
        return h

    def bprop(self, h, igrads, first_layer=False):
        #identical to Relu implementation in layers.py
        #input from fully connected layer
        if igrads.ndim == 2:
            igrads = igrads.reshape(h.shape)
        deltas = (h > 0)*igrads
        ___, ograds = super(ConvRelu, self).bprop(h=None, igrads=deltas, first_layer=first_layer)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvRelu.bprop_cost method not implemented')

    def get_name(self):
        return 'convrelu'


if __name__ == "__main__":
    #tests via doctest https://docs.python.org/2/library/doctest.html#module-doctest
    #to see detailed ouptut run as:
    #python conv.py -v
    doctest.testmod()