#L1 test
from support import *
from mlp.schedulers import LearningRateFixed
from mlp.optimisers import SGDOptimiser

LearningRate = 0.7
Num_epoches = 30
l1_weights = [10, 1, 0.1, 0.01, 0.001]
tsk2_1_jobs  = []
for l1_w in l1_weights:
    lr_scheduler = LearningRateFixed(LearningRate, Num_epoches)
    tsk2_1_jobs.append(
        {
            "model": create_one_hid_model(),
            "label": "l1_w={}".format(l1_w),
            "optimiser": SGDOptimiser(lr_scheduler, l1_weight=l1_w)
        }
    )

tsk2_1_stats = get_models_statistics(tsk2_1_jobs)

