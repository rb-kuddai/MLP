import matplotlib.pyplot as plt
import numpy as np

#plot best regularizers
def layer_hist(layers, labels):
    f, axis = plt.subplots(1, len(layers), sharey=True, figsize=(15,4))
    for nth, label, layer in zip(range(len(layers)), labels, layers):
        ax = axis[nth]
        W, b = layer.get_params()
        w_mag = list(np.abs(W.reshape(-1)))
        ax.set_title(label)
        ax.hist(w_mag, bins=35)
        ax.set_ylabel("number of weights")
        ax.set_xlabel("weight magnitute")

    plt.show()
"""
ml_l1, lb_l1 = tsk2_1_jobs[2]["model"], tsk2_1_jobs[2]["label"]
ml_l2, lb_l2 = tsk2_2_jobs[2]["model"], tsk2_2_jobs[2]["label"]
ml_nr, lb_nr = tsk2_3_jobs[0]["model"], tsk2_3_jobs[0]["label"]

for i in range(2):
    ad = " layer {}".format(i)
    layer_hist([ml_l1.layers[i], ml_l2.layers[i], ml_nr.layers[i]], [lb_l1 + ad, lb_l2 + ad, lb_nr + ad])
"""