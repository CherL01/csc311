from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(321)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for user, q_id, correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        if np.isnan(correct):
            continue
        g = sigmoid(theta[user] - beta[q_id])
        log_lklihood += correct*np.log(g) + (1-correct)*np.log(1-g)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    d_theta = np.zeros_like(theta)
    d_beta = np.zeros_like(beta)
    for user, q_id, correct in zip(data['user_id'], data['question_id'], data['is_correct']):
        if np.isnan(correct):
            continue
        g = sigmoid(theta[user] - beta[q_id])
        d_g = g * (1 - g)
        d_theta[user] += correct * (1 / g) * d_g + (1-correct) * (1 / (1-g)) * (-d_g)
        d_beta[q_id] += correct * (1 / g) * d_g * - 1 + (1-correct) * (1 / (1-g)) * (-d_g) * - 1

    theta = theta + lr * d_theta
    beta = beta + lr * d_beta
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(set(data['user_id'])))
    beta = np.zeros(len(set(data['question_id'])))

    neg_lld_lst = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        neg_lld_lst.append(neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    
    lr = 0.01
    num_iter = 60
    # print(len(train_data["user_id"]))
    theta, beta, val_acc_lst, neg_lld_lst = irt(train_data, val_data, lr, num_iter)

    x = [i for i in range(1, num_iter+1)]
    plt.plot(x, neg_lld_lst)
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Negative Log-Likelihood vs. Iterations')
    plt.savefig('/Users/cherry.lian/Desktop/CSC311/CSC311_Project/csc311/starter_code/part_a/q2_b_nlld.png')

    test_acc = evaluate(test_data, theta, beta)
    print('validation accuracy: ', val_acc_lst[-1])
    print('test accuracy: ', test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    
    js = random.sample(range(len(beta)), 3)

    probabilities = []
    theta_range = np.sort(theta)
    # theta_range = np.linspace(np.min(theta), np.max(theta), num=100)
    for j in range(len(js)):
        prob = sigmoid(theta_range - beta[js[j]])
        probabilities.append(prob)

    plt.figure()
    for p, j in zip(probabilities, js):
        plt.plot(theta_range, p, label=f"j={j}")
    plt.legend()
    plt.xlabel('Theta')
    plt.ylabel('Probaility')
    plt.title('Probability as a Funtion of Theta')
    plt.savefig('/Users/cherry.lian/Desktop/CSC311/CSC311_Project/csc311/starter_code/part_a/q2_d_prob.png')
    

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
