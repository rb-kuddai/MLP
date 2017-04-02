from mlp.schedulers import LearningRateExponential, LearningRateReciprocal
from support import *

schd_tasks  = [
    {
    "model": create_one_hid_model(),
    "label": "Exponential r=5 lr0=0.7",
    "lr_scheduler": LearningRateExponential(0.7, 30, 5.0)
    },
    {
    "model": create_one_hid_model(),
    "label": "Reciprocal r=5 lr0=0.7 c=0.5",
    "lr_scheduler": LearningRateReciprocal(0.7, 30, 5.0, 0.5)
    },
    {
    "model": create_one_hid_model(),
    "label": "Reciprocal r=5 lr0=0.7 c=1",
    "lr_scheduler": LearningRateReciprocal(0.7, 30, 5.0, 1.0)
    },
    {
    "model": create_one_hid_model(),
    "label": "Reciprocal r=5 lr0=0.7 c=2",
    "lr_scheduler": LearningRateReciprocal(0.7, 30, 5.0, 2.0)
    }
]

schd_stats = get_models_statistics(schd_tasks)
