from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import torch


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


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        act = nn.Sigmoid()
        out = act(self.h(act(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, device = torch.device("cpu")):
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
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()
    model.to(device)

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # num_student = train_data.shape[0]

    dataloader = DataLoader(zero_train_data, batch_size=1, shuffle=False, num_workers=4)
    dl2 = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for data, nan_mask in zip(dataloader, dl2):
            inputs = data.to(device)
            target = inputs.clone().to(device)

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(nan_mask.to(device))
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # loss = torch.sum((output - target) ** 2.) + lamb/2 * (model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, device=device)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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
    k = [10, 50, 100, 200, 500]
    model = AutoEncoder

    num_questions = train_matrix.shape[1]

    # Set optimization hyperparameters.
    lr = 0.03
    num_epoch = 30
    lamb = 0
    for k_choice in k:
        train(model(num_question=num_questions, k=k_choice), lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch, 
          device=torch.device("cuda:0")
          )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
