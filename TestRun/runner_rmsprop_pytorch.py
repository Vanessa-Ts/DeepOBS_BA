import numpy as np

from torch.optim import rmsprop, SGD, Adam
from deepobs import pytorch as pt
from deepobs import config

DATA_DIR = "../data_deepobs"


def run_testproblem_oneshot(optimizer, hyperparameter, testproblem, defaults):
    runner = pt.runners.StandardRunner(optimizer, hyperparameter)
    batch_size = defaults[testproblem]["batch_size"]
    epochs = defaults[testproblem]["num_epochs"]
    runner.run(testproblem=testproblem, hyperparams={'learning_rate': hyperparameter["lr"]["default"]},
               batch_size=batch_size, num_epochs=40, random_seed=42, data_dir=DATA_DIR,
               l2_reg=None, no_logs=None, train_log_interval=None, print_train_iter=True, tb_log=None, tb_log_dir=None,
               skip_if_exists=False, eval_interval=5)


def init_default_problem_params(testproblem):
    defaults = {}
    tesproblem_default = config.get_testproblem_default_setting(testproblem)
    defaults[testproblem] = tesproblem_default
    return defaults


if __name__ == '__main__':
    testproblem = "fmnist_dcgan"
    defaults = init_default_problem_params(testproblem)

    optimizer_class = rmsprop
    hyperparams = {
        "lr": {"type": float, "default": 0.01},
        "alpha": {"type": float, "default": 0.99},
        "eps": {"type": float, "default": 1e-08},
        "weight_decay": {"type": float, "default": 0},
        "momentum": {"type": float, "default": 0},
        "centered": {"type": bool, "default": False},
    }
    run_testproblem_oneshot(optimizer_class, hyperparams, testproblem, defaults)
    print(defaults)
