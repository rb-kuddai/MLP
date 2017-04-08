 # Description
[Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/) is a course devoted to neural networks (NN). There we have gradually built our own small framework for sequential NN models in numpy. The course consisted of two courseworks. We evaluated both of them on MNIST classification dataset.  The final reports along with the relevant code can be seen in jupyter notebooks.

In the first task ([03_MLP_Coursework1.ipynb](https://github.com/rb-kuddai/MLP/blob/master/03_MLP_Coursework1.ipynb)) it was required to implement layers (forward, backward passes and error functions for final layers) for standard Multiple Layer Perceptron (MLP). Additionally, for ease of experimation I have created a model wrapper ([support.py](https://github.com/rb-kuddai/MLP_RU/blob/master/mlpractical/support.py)) which allows to put tasks in a queue with exception handling, set and/or serialise model parameters as dictionaries and save the best models (features similar to those of scikit package). It helped greatly in the next task. The final mark for the first task was 90/100.

The second task ([s1569105-07_MLP_Coursework2.ipynb](https://github.com/rb-kuddai/MLP/blob/master/s1569105-07_MLP_Coursework2.ipynb)) was devoted to regularisation, autoencoders and Convolution Neural Networks (CNN). The main problem in this task was that the straightforward implementation of a convolution operator in native Python is too slow (we were allowed to use only Python and Cython). Cython alleviated speed issues to some extent but it was still not enough. The best results have been achieved through unfolding a convolution operator as huge 2-D matrices products. In numpy, such dot product can be accelerated greatly through libraries like OpenBLAS and the usage of several CPU cores. My implementation is located here [conv.py](https://github.com/rb-kuddai/MLP/blob/master/mlpractical/mlp/conv.py), (for comparison the original file which was provided to us [conv.py](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2015-6/master/mlp/conv.py)). This file is actually quite compact and most of it is taken by doctests. Also, during debugging the Finite Difference Method helped greatly to check correctness of gradients, as well a manual test from this publication:
> Chellapilla, Kumar, Sidd Puri, and Patrice Simard.  "High performance convolutional neural networks for document processing."  Tenth International Workshop on Frontiers in Handwriting Recognition. Suvisoft, 2006 

The final boost in performance was achieved by noticing a small design flaw in the original code provided to us. The code for the container which held NN layers was originally calculating a backpropagation error for the first hidden layer (the one which was connected to input X). But it was pointless as there were no hidden layers before the very first one and CPU cycles were wasted. It played a crucial part in the overall training time as due to hardware limitations our networks were quite shallow. That is why after fixing I have gained almost 2 times performance boost. This code [layers.py](https://github.com/rb-kuddai/MLP/blob/master/mlpractical/mlp/layers.py) can be seen in (class MLP_fast, line 194). Along with dataset augmentation via  simple affine transformations (scale, shifts, rotations), this boost has allowed to attain 99.21% +- 0.0885% accuracy on the test data set (10000 images). The final mark was 100/100.


