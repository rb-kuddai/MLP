from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mlp.dataset import MNISTDataProvider

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

def test():
    mnist_dp = MNISTDataProvider(dset='valid', batch_size=4, max_num_examples=4, randomize=False)
    for batch in mnist_dp:
        features, targets = batch
        num_imgs = features.shape[0]
        imgs = features.reshape(num_imgs, 28, 28)

        shf_img = ndimage.interpolation.shift(imgs, shift=(0, 10, 10))
        tot_imgs = np.concatenate((imgs, shf_img), axis=0)
        show_mnist_images(tot_imgs, 4, 3, ["original"] * 4 + ["shifted"] * 4)