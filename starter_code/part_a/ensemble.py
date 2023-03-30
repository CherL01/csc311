from utils import *
from neural_network import AutoEncoder, load_data, train

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import torch

def split_data(data, zero_data, splits, ):
    # function to split the training data for bagging
    split_idxs = np.random.choice(data.shape[0], size=(splits, data.shape[0]))
    split_datas = []
    split_zero_datas = []
    for split_idx in split_idxs:
        split_datas.append(data[split_idx, :])
        split_zero_datas.append(zero_data[split_idx, :])

    return split_datas, split_zero_datas

def evaluate_ensemble(models, train_data, valid_data, device=torch.device("cpu")):
    """ Evaluate the valid_data on a list of models.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    total = 0
    correct = 0
    outputs = []
    for model in models:
        model.eval()
        model.to(device)

        inputs = Variable(train_data).to(device)
        output = model(inputs).to(device)
        outputs.append(output)

    output = torch.mean((torch.stack(outputs)), dim=0)

    valid_guesses = output[valid_data["user_id"], valid_data["question_id"]]
    guess = (valid_guesses >= 0.5)
    correct = torch.sum(guess == torch.tensor(valid_data["is_correct"], device=device)).item()

    total = guess.shape[0]
    
    return correct / float(total)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    splits = 3

    train_matrix_split, zero_train_matrix_split = split_data(train_matrix, zero_train_matrix, splits)    
    
    # train 3 autoencoders
    device = torch.device("cuda:0")

    num_questions = train_matrix.shape[1]

    lr = 0.017
    num_epoch = 15
    lamb = 0
    k = 10

    models = [AutoEncoder(num_questions, k) for _ in range(splits)]

    top_accs = []
    for i, (zero_train_mat, train_mat) in enumerate(zip(zero_train_matrix_split, train_matrix_split)):
        train_costs, val_accs = train(models[i], lr, lamb, train_mat, zero_train_mat,
          valid_data, num_epoch, 
          device=device
          )
        top_accs.append(val_accs[-1])

    acc = evaluate_ensemble(models, zero_train_matrix, valid_data, device=device)
    print(acc)

    test_acc = evaluate_ensemble(models, zero_train_matrix, test_data, device=device)
    print(test_acc)


if __name__ == "__main__":
    main()