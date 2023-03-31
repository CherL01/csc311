import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from utils import *


class StudentData(Dataset):
    
    def __init__(self, zero_train_matrix, train_matrix, data_path="../data") -> None:
        self.zero_train_matrix = zero_train_matrix
        self.train_matrix = train_matrix

        df = pd.read_csv(os.path.join(data_path, "student_meta.csv"), index_col="user_id")
        df["data_of_birth"] = df["data_of_birth"].fillna(value="2005").apply(func=lambda x: int(x.split("-")[0]))
        df["data_of_birth"] = 2020 - df["data_of_birth"]

        df["premium_pupil"] = df["premium_pupil"].fillna(value=0.0).apply(func=lambda x: int(x))
        df["gender"] = df["gender"].fillna(value=0).apply(func=lambda x: int(x))

        self.student_df = df

        self.normalize_meta_data()

    def __len__(self):
        return self.train_matrix.shape[0]

    def normalize_meta_data(self):
        for col in self.student_df.columns:
            if col == "data_of_birth":
                df = self.student_df[col]
                self.student_df[col] = (df - np.mean(df)) / np.std(df)

    def __getitem__(self, index) -> torch.Tensor:
        
        student_meta_data = torch.tensor(self.student_df.loc[[index], ["gender", "premium_pupil"]].to_numpy())

        student_meta_data = F.one_hot(student_meta_data, num_classes=3).reshape(-1)

        student_meta_data = torch.concat((student_meta_data, torch.tensor(self.student_df.loc[[index], ["data_of_birth"]].to_numpy()).reshape(-1)))

        return self.zero_train_matrix[index], self.train_matrix[index], student_meta_data.type(torch.float32)

if __name__ == "__main__":

    dataset = StudentData([0], [0])
    print(dataset[0])
