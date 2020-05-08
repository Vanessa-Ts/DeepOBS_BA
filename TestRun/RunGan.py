from deepobs.pytorch.datasets.fmnist import fmnist  # Import the data loading module of DeepOBS
from deepobs import pytorch as pt
from deepobs import config
from deepobs.pytorch.testproblems import fmnist_dcgan
from deepobs.pytorch.testproblems import testproblem, testproblems_utils, testproblems_modules
from deepobs.pytorch.datasets import dataset, datasets_utils


DATA_DIR = "../data_deepobs"

data = fmnist(batch_size=128)  # Create an instance of the FMNIST Data class (which is a subclass of the Data Set class of DeepOBS), using for example 64 as the batch size.


next_batch = next(iter(data._train_dataloader))  # get the next batch of the training data set. If you replace '_train_dataloader', with '_test_dataloader' you would get a batch of the test data set and so on.

print(len(next_batch))
print(len(next_batch[0]))


testproblem = fmnist_dcgan(batch_size=128)
testproblem.set_up()

testproblem.train_init_op() # use training data set



