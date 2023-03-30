import numpy as np
import torch

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import StudentData
from models import BetterAutoEncoder

from utils import *

def train(model, lr, lamb, train_dataloader, valid_data, num_epoch, device = torch.device("cpu")):
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
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)

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

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

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
        

    inputs = Variable(train_data).to(device)
    output = model(inputs).to(device)
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
    k = 100
    model = AutoEncoder

    num_questions = train_matrix.shape[1]

    # Set optimization hyperparameters.
    lr = 0.05
    num_epoch = 15
    # lamb = [0.001, 0.01, 0.1, 1]
    lamb = [0.001]
    top_accs = []
    for lamb_choice in lamb:
        m = model(num_question=num_questions, k=k)
        train_costs, val_accs = train(m, lr, lamb_choice, train_matrix, zero_train_matrix,
          valid_data, num_epoch, 
          device=torch.device("cuda:0")
          )
        top_accs.append(val_accs[-1])
    
    # plt.plot(range(1, num_epoch+1, 1), val_accs)
    # plt.title("Validation Accuracy vs Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.savefig("q3_d_val.png")
    # plt.clf()

    # plt.plot(range(1, num_epoch+1, 1), train_costs)
    # plt.title("Training Loss vs Epochs")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.savefig("q3_d_train.png")
    # plt.plot(lamb, top_accs)
    # plt.title("Validation Accuracy vs Regularization")
    # plt.xlabel("Lambda")
    # plt.ylabel("Validation Accuracy")
    # plt.savefig("q3_e.png")
    test_acc = evaluate(m, zero_train_matrix, test_data, device=torch.device("cuda:0"))

    print(f"Test accuracy: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

