# %load my_conv_tests.py
import numpy
import time
from mlp.utils import test_conv_linear_fprop
from mlp.utils import test_conv_linear_bprop
from mlp.utils import test_conv_linear_pgrads
from mlp.utils import verify_layer_gradient
from mlp.conv import ConvLinear, ConvSigmoid, ConvRelu, my_conv_fwd2, my_conv_bck2, my_conv_grad2


#Linear layer basic tests:
#put 0, 0 because tests will put W, b internally

#Test forward propagation
layer = ConvLinear(0, 0)
test_conv_linear_fprop(layer)

#Test backward propagation
layer = ConvLinear(0, 0)
test_conv_linear_bprop(layer)

#Test gradients
layer = ConvLinear(0, 0)
test_conv_linear_pgrads(layer)

#Test forward propagation for 2 version of convolution
#conv_fwd=my_conv_fwd,
#conv_bck=my_conv_bck,
#conv_grad=my_conv_grad
layer = ConvLinear(0, 0, conv_fwd=my_conv_fwd2, conv_bck=my_conv_bck2, conv_grad=my_conv_grad2)
test_conv_linear_fprop(layer)

#Test backward propagation for 2 version of convolution
layer = ConvLinear(0, 0, conv_fwd=my_conv_fwd2, conv_bck=my_conv_bck2, conv_grad=my_conv_grad2)
test_conv_linear_bprop(layer)

#Test gradients for 2 version of convolution
layer = ConvLinear(0, 0, conv_fwd=my_conv_fwd2, conv_bck=my_conv_bck2, conv_grad=my_conv_grad2)
test_conv_linear_pgrads(layer)

#Gradient tests

#Test linear layer with non-standard strides (2, 2)
inputs = numpy.arange(128).reshape((4, 2, 4, 4)).astype(numpy.float)
convlinear = ConvLinear(num_inp_feat_maps = 2,
                   num_out_feat_maps = 2,
                   image_shape = (4,4),
                   kernel_shape = (2,2),
                   stride = (2,2))
verify_layer_gradient(convlinear, inputs)

#Test ConvSigmoid gradients
inputs = numpy.arange(128).reshape((4, 2, 4, 4)).astype(numpy.float)
#scalling because exponent of 128 it is just too large
inputs /= 32
convsigmoid = ConvSigmoid(num_inp_feat_maps = 2,
                   num_out_feat_maps = 2,
                   image_shape = (4,4),
                   kernel_shape = (2,2),
                   stride = (2,2))
verify_layer_gradient(convsigmoid, inputs)


#Test ConvRelu gradients
inputs = numpy.arange(128).reshape((4, 2, 4, 4)).astype(numpy.float)
convrelu = ConvRelu(num_inp_feat_maps = 2,
                   num_out_feat_maps = 2,
                   image_shape = (4,4),
                   kernel_shape = (2,2),
                   stride = (2,2))
verify_layer_gradient(convrelu, inputs)


#test for non-standard strides (2,2)
krl = numpy.ones((1,1,2,2))
img = numpy.ones((1,1,4,4))
correct_output_img = numpy.array([[[[4, 4],
                                    [4, 4]]]])
layer = ConvLinear(num_inp_feat_maps = 1,
                   num_out_feat_maps = 1,
                   image_shape=(4, 4),
                   kernel_shape=(2, 2),
                   stride=(2,2))
layer.set_params([krl, numpy.array([0, 0])])
assert (layer.fprop(img) == correct_output_img).all()

#The test is taken from the paper:
#Chellapilla, Kumar, Sidd Puri, and Patrice Simard.
#"High performance convolutional neural networks for document processing."
#Tenth International Workshop on Frontiers in Handwriting Recognition. Suvisoft, 2006.

image = numpy.empty((1, 3, 3, 3), dtype=numpy.int64)
image[0, 0, :, :] = numpy.array([[1, 2, 0], [1, 1, 3], [0, 2, 2]])
image[0, 1, :, :] = numpy.array([[0, 2, 1], [0, 3, 2], [1, 1, 0]])
image[0, 2, :, :] = numpy.array([[1, 2, 1], [0, 1, 3], [3, 3, 2]])

kernel = numpy.empty((3, 2, 2, 2), dtype=numpy.int64)
kernel[0, 0, :, :] = numpy.array([[1, 1], [2, 2]])
kernel[1, 0, :, :] = numpy.array([[1, 1], [1, 1]])
kernel[2, 0, :, :] = numpy.array([[0, 1], [1, 0]])

kernel[0, 1, :, :] = numpy.array([[1, 0], [0, 1]])
kernel[1, 1, :, :] = numpy.array([[2, 1], [2, 1]])
kernel[2, 1, :, :] = numpy.array([[1, 2], [2, 0]])

img = image.swapaxes(2, 3)
krl = kernel.swapaxes(2, 3)
layer = ConvLinear(num_inp_feat_maps = 3,
                   num_out_feat_maps = 2,
                   image_shape=(3, 3),
                   kernel_shape=(2, 2))
layer.set_params([krl, numpy.array([0, 0])])
correct_output_img = numpy.array([[[[14, 15],
                                    [20, 24]],
                                   [[12, 17],
                                    [24, 26]]]])

assert (layer.fprop(img) == correct_output_img).all()

N = 100000

tstart = time.clock()
for i in xrange(N):
    layer.fprop(img)
tstop = time.clock()
print "2x2 by 2x2 matrices convolutions (my_conv_fwd) {0}".format(tstop - tstart)

layer = ConvLinear(num_inp_feat_maps = 3,
                   num_out_feat_maps = 2,
                   image_shape=(3, 3),
                   kernel_shape=(2, 2),
                   conv_fwd=my_conv_fwd2,
                   conv_bck=my_conv_bck2,
                   conv_grad=my_conv_grad2)

layer.set_params([krl, numpy.array([0, 0])])

tstart = time.clock()
for i in xrange(N):
    layer.fprop(img)
tstop = time.clock()
print "tensor product convolutions (my_conv_fwd2) {0}".format(tstop - tstart)

