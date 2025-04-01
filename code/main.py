from utils import load_data2
from train_test import train_and_evaluate_on_multiple_datasets
from model import AemlpModel
import numpy as np

if __name__ == "__main__":
    data_base_folder = '../data_5k'
    datasets =['LGG1', 'LGG2', 'LGG3', 'LGG4', 'LGG5']
    batch_size = 34
    num_epochs = 300
    ae_input = [2000, 2000, 548]
    ae_class = [2, 2, 2]
    input_dims = [180, 180, 180]#[180, 180, 180]
    num_classes = 2
    lambda_a = [208,1,1,1,197]

    train_and_evaluate_on_multiple_datasets(data_base_folder,datasets, batch_size,
                                            num_epochs, ae_input, ae_class,
                                            input_dims, num_classes,lambda_a)
