import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from utils import *

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class StudentData(Dataset):
    
    def __init__(self, zero_train_matrix, train_matrix, data_path="../data") -> None:
        self.zero_train_matrix = zero_train_matrix
        self.train_matrix = train_matrix

        self.student_df = pd.read_csv("../data/student_meta.csv")
        self.student_df



    def __len__(self):
        return self.train_matrix.shape[0]

    def normalize_meta_data(self):
        for col in self.student_df.columns:
            if col == "user_id":
                self.student_df[col]

if __name__ == "__main__":
    df = pd.read_csv("../data/student_meta.csv", index_col="user_id")
    df["data_of_birth"] = df["data_of_birth"].fillna(value="2005").apply(func=lambda x: int(x.split("-")[0]))
    df["data_of_birth"] = 2020 - df["data_of_birth"]

    df["premium_pupil"] = df["premium_pupil"].fillna(value=0.0).apply(func=lambda x: int(x))
    df["gender"] = df["gender"].fillna(value=0).apply(func=lambda x: int(x))

    print(df.head())