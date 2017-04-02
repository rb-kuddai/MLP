from mlp.dataset import  MNISTDataProviderNoisy
#init test
"""
print "init with noise"
train_dp_rt = MNISTDataProviderNoisy(dset='train',
                                      num_of_gen_data=50000,
                                      noise_param=0.1,
                                      noise_fun=MNISTDataProviderNoisy.apply_noise,
                                      batch_size=100,
                                      max_num_batches=-10)
print "init with rotations"
train_dp_ns = MNISTDataProviderNoisy(dset='train',
                                      num_of_gen_data=50000,
                                      noise_param=10.0,
                                      noise_fun=MNISTDataProviderNoisy.apply_rotation,
                                      batch_size=100,
                                      max_num_batches=-10)
print "init with shifts"
train_dp_sft = MNISTDataProviderNoisy(dset='train',
                                      num_of_gen_data=50000,
                                      noise_param=10.0,
                                      noise_fun=MNISTDataProviderNoisy.apply_shift,
                                      batch_size=100,
                                      max_num_batches=-10)
"""

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mlp.dataset import MNISTDataProvider, MNISTDataProviderNoisy

def show_mnist_image(img):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.imshow(img, cmap=cm.Greys_r)
    plt.show()

def show_mnist_images(imgs, imgs_per_line, img_size, labels=None):
    n = imgs.shape[0]
    form = (imgs_per_line, n/imgs_per_line + (n%imgs_per_line != 0))
    fig = plt.figure(figsize=(form[0] * img_size, form[1] * img_size))
    gs = gridspec.GridSpec(form[1], form[0])

    for img_id, img in enumerate(imgs):
        row_id, column_id = img_id / form[0], img_id % form[0]
        a = 5
        ax = fig.add_subplot(gs[row_id, column_id])
        #remove clumsy numbers on both axis
        if labels is not None:
            ax.set_title(labels[img_id])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #convert from mnist image format
        ax.imshow(img, cmap=cm.Greys_r)
    plt.show()

def show_noise():
    mnist_dp = MNISTDataProvider(dset='valid', batch_size=1, max_num_examples=1, randomize=False)
    for batch in mnist_dp:
        features, targets = batch
        noise_ratios = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0]#[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.8]
        #apply_ns = MNISTDataProviderNoisy.apply_noise
        apply_ns = MNISTDataProviderNoisy.apply_rotation
        imgs_ns = np.array([apply_ns(features[0], nr) for nr in noise_ratios])
        tot_imgs = np.concatenate((features, imgs_ns), axis=0)
        tot_imgs = tot_imgs.reshape(tot_imgs.shape[0], 28, 28)
        show_mnist_images(tot_imgs, 4, 3, ["original"] + ["noise ratio " + str(nr) for nr in noise_ratios])

show_noise()