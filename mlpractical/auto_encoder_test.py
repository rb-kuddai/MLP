from mlp.optimisers import SGDOptimiser
from support import train_dp_flat, valid_dp_flat, create_one_hid_model, rng
from mlp.costs import CECost
from mlp.layers import MLP, Softmax, Sigmoid
from mlp.schedulers import LearningRateFixed
from noise_tries import show_mnist_images
import numpy as np
from mlp.dataset import MNISTDataProvider
from mlp.optimisers import Optimiser

tsk3_2_optimiser = Optimiser()
tsk3_2_model = create_one_hid_model()

def tsk_3_2_draw_handler(cur_layer_id, cur_model, get_inputs):
    if cur_layer_id != 0:
        return
    mnist_dp = MNISTDataProvider(dset='valid', batch_size=4, max_num_examples=4, randomize=False)
    for batch in mnist_dp:
        features, targets = batch

        inputs, pure = get_inputs(features)
        output_dc = cur_model.fprop(inputs)
        num_imgs = features.shape[0]
        imgs = features.reshape(num_imgs, 28, 28)
        #images noisy
        imgs_ns = inputs.reshape(num_imgs, 28, 28)
        #images decoded
        imgs_dc = output_dc.reshape(num_imgs, 28, 28)
        tot_imgs = np.concatenate((imgs, imgs_ns, imgs_dc), axis=0)
        show_mnist_images(tot_imgs, 4, 3, ["original"] * 4 + ["with noise"] * 4 + ["decoded"] * 4)

tsk3_2_stats = tsk3_2_optimiser.pretrain_masking(tsk3_2_model, train_dp_flat, 20, 0.05, 0.9, draw_handler=tsk_3_2_draw_handler)



#optimiser.train(model, train_dp_flat, valid_dp_flat)
