import numpy
from mlp.utils import verify_layer_gradient
from mlp.maxpooling import ConvMaxPool2D, ConvMaxPool2D_Cython

#test non standard stride shapes
#here pool_shape is (2,2) but stride shape is (1,1)
inputs = numpy.array([[[[ 0,  1,  2],
                        [ 3,  4, 45],
                        [ 6,  7,  8]]]])
#results in overlapping windows
convmaxpool2d = ConvMaxPool2D(num_feat_maps=1,
                              conv_shape=(3,3),
                              pool_shape=(2,2),
                              pool_stride=(1,1))

correct_output = numpy.array([[[[ 4, 45],
                                [ 7, 45]]]])

assert (convmaxpool2d.fprop(inputs) == correct_output).all()

#Test for forward and backward pass
#note that output of the fprop is given as igrads for bprop
#for easy interpretation
inputs = numpy.array([[[[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11],
                        [12, 13, 14, 15]]]])

convmaxpool2d = ConvMaxPool2D(num_feat_maps=1,
                              conv_shape=(4,4),
                              pool_shape=(2,2),
                              pool_stride=(2,2))

correct_output = numpy.array([[[[ 5,  7],
                                [13, 15]]]])

assert (convmaxpool2d.fprop(inputs) == correct_output).all()

correct_ograds = numpy.array([[[[  0.,   0.,   0.,   0.],
                                  [  0.,   5.,   0.,   7.],
                                  [  0.,   0.,   0.,   0.],
                                  [  0.,  13.,   0.,  15.]]]])

assert (convmaxpool2d.bprop(inputs, correct_output)[1] == correct_ograds).all()

#Test maxpooling.py
inputs = numpy.arange(16).reshape((1, 1, 4, 4)).astype(numpy.float)
convmaxpool2d = ConvMaxPool2D(num_feat_maps=1,
                              conv_shape=(4,4),
                              pool_shape=(2,2),
                              pool_stride=(2,2))

verify_layer_gradient(convmaxpool2d, inputs)

"""
inputs = numpy.arange(16).reshape((1, 1, 4, 4)).astype(numpy.float)
convmaxpool2d_cython = ConvMaxPool2D_Cython(num_feat_maps=1,
                                            conv_shape=(4,4),
                                            pool_shape=(2,2),
                                            pool_stride=(2,2))
#I am using floats so the difference is neglegitible
verify_layer_gradient(convmaxpool2d_cython, inputs)
"""