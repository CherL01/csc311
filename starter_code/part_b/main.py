import numpy as np
import torch

import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import StudentData
from models import BetterAutoEncoder

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

def train(model, lr, lamb, train_dataloader, valid_data, num_epoch, step_size = 5, device = torch.device("cpu")):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    train_losses = []
    val_accs = []

    # Tell PyTorch you are training the model.
    model.train()
    model.to(device)

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamb)
    # num_student = train_data.shape[0]
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.5)

    loss_fn = nn.MSELoss()

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for data, nan_mask, meta_data in train_dataloader:
            inputs = data.to(device)
            meta_inputs = meta_data.to(device)
            target = inputs.clone().to(device)

            optimizer.zero_grad()
            output = model(inputs, meta_inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(nan_mask.to(device))
            target[nan_mask] = output[nan_mask]

            loss = loss_fn(output, target)
            # print(loss)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        scheduler.step()
        valid_acc = evaluate(model, train_dataloader, valid_data, device=device)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
        train_losses.append(train_loss)
        val_accs.append(valid_acc)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return train_losses, val_accs


def evaluate(model, train_data, valid_data, device=torch.device("cpu")):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    zeroed_data = []
    meta_data = []
    for z_d, d, m_d in train_data:
        zeroed_data.append(z_d)
        meta_data.append(m_d)

    zeroed_data = torch.concatenate(zeroed_data, dim=0).to(device)
    meta_data = torch.concatenate(meta_data, dim=0).to(device)

    output = model(zeroed_data, meta_data)
    valid_guesses = output[valid_data["user_id"], valid_data["question_id"]]
    guess = (valid_guesses >= 0.5)
    correct = torch.sum(guess == torch.tensor(valid_data["is_correct"], device=device)).item()

    total = guess.shape[0]
    
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10, 50, 100, 200, 500]
    num_questions = train_matrix.shape[1]

    dataset = StudentData(zero_train_matrix=zero_train_matrix, train_matrix=train_matrix)
    

    # Set optimization hyperparameters.
    lr = [0.1, 0.05, 0.01]
    num_epoch = 50
    lamb = [0.0, 0.001, 0.0001]
    steps = [5, 10, 25]
    top_accs = []
    test_accs = []
    for step_choice in steps:
        for lr_choice in lr:
            for lamb_choice in lamb:
                for k_choice in k:
                    model = BetterAutoEncoder(num_question=num_questions, meta_features=7)
                    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
                    train_costs, val_accs = train(model, lr_choice, lamb_choice, dataloader,
                    valid_data, num_epoch, step_size=step_choice,
                    device=torch.device("cuda:0")
                    )
                    top_accs.append(val_accs[-1])
                    test_acc = evaluate(model, dataloader, test_data, device=torch.device("cuda:0"))
                    test_accs.append(test_acc)
            
                    with open("log.txt", "a") as f:
                        f.write(",".join([str(k_choice), str(lamb_choice), str(lr_choice), str(step_choice), str(val_accs[-1]), str(train_costs[-1]), str(test_acc)]) + "\n")
    # plt.plot(range(1, num_epoch+1, 1), val_accs)
    # plt.title("Validation Accuracy vs Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.savefig("part_b_val.png")
    # plt.clf()

    # plt.plot(range(1, num_epoch+1, 1), train_costs)
    # plt.title("Training Loss vs Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.savefig("part_b_train.png")
    # plt.plot(lamb, top_accs)
    # plt.title("Validation Accuracy vs Regularization")
    # plt.xlabel("Lambda")
    # plt.ylabel("Validation Accuracy")
    # plt.savefig("q3_e.png")
    print(top_accs)
    print(test_accs)
    test_acc = evaluate(model, dataloader, test_data, device=torch.device("cuda:0"))
    print(f"Test accuracy: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

