from support import *
from mlp.dataset import MNISTDataProviderNoisy
LearningRate = 0.7
Num_epoches = 30
num_of_gen_data = 50000
noise_ratios = [0.01] #, 0.01, 0.05, 0.10, 0.15]
tsk4_1_jobs  = []
print "start tasks"
for ns in noise_ratios:
    train_dp = MNISTDataProviderNoisy(dset='train', 
                                      num_of_gen_data=num_of_gen_data,
                                      noise_param=ns,
                                      batch_size=100,
                                      max_num_batches=-10)

    tsk4_1_jobs.append(
        {
            "model": create_one_hid_model(),
            "label": "noise ratio={}".format(ns),
            "lr_rate": LearningRate,
            "max_epochs": Num_epoches,
            "train_dp": train_dp,
            "save": True
        }
    )

tsk4_1_stats = get_models_statistics(tsk4_1_jobs)