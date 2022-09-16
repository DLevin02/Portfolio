## Written by: Drew Levin

import numpy as np


def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')

# test_labels might be different for you
test_labels = [9, 3]
indices = np.where(np.isin(y_train, test_labels))[0]

x = x_train[indices]
y = y_train[indices]

y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1

## number of units in the hidden layer
h = 28

# number of units in the input layer, i.e., 784
m = x.shape[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(o):
    return o * (1 - o)


# adjust alpha and number of epochs by yourself
alpha = .1
num_epochs = 1
num_train = len(y)


def nnet(train_x, train_y, alpha, num_epochs, num_train):
    w1 = np.random.uniform(low=-1, high=1, size=(m, h))
    w2 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b1 = np.random.uniform(low=-1, high=1, size=(h, 1))
    b2 = np.random.uniform(low=-1, high=1, size=(1, 1))

    # set a large number as the initial cost to be compared with in the 1st iteration
    loss_previous = 10e10

    for epoch in range(1, num_epochs + 1):
        # shuffle the dataset
        train_index = np.arange(num_train)
        np.random.shuffle(train_index)
        for i in train_index:
            # a1 will be of the dimension of 28 * 1
            a1 = sigmoid(w1.T @ train_x[i, :].reshape(-1, 1) + b1)
            # a2 is a 1*1 matrix
            a2 = sigmoid(w2.T @ a1 + b2)

            # dCdw1 will be a 28 * 784 matrix
            dCdw1 = (a2 - train_y[i]) * sigmoid_derivative(a2) * w2 * sigmoid_derivative(a1) * (
                train_x[i, :].reshape(1, -1))
            # dCdb1 will be a 28 * 1 matrix
            dCdb1 = (a2 - train_y[i]) * sigmoid_derivative(a2) * w2 * sigmoid_derivative(a1)
            # dCdw2 will be a a 28 * 1 matrix
            dCdw2 = (a2 - train_y[i]) * sigmoid_derivative(a2) * a1
            # dCdb2 will be a 1*1 matrix
            dCdb2 = (a2 - train_y[i]) * sigmoid_derivative(a2)

            # update w1, b1, w2, b2
            w1 = w1 - alpha * dCdw1.T
            b1 = b1 - alpha * dCdb1
            w2 = w2 - alpha * dCdw2
            b2 = b2 - alpha * dCdb2

        # the output of the hidden layer will be a num_train * 28 matrix
        out_h = sigmoid(train_x @ w1 + b1.T)
        # the output of the output layer will be a num_train * 1 matrix
        out_o = sigmoid(out_h @ w2 + b2)

        loss = .5 * np.sum(np.square(y.reshape(-1, 1) - out_o))
        loss_reduction = loss_previous - loss
        loss_previous = loss
        correct = sum((out_o > .5).astype(int) == y.reshape(-1, 1))
        accuracy = (correct / num_train)[0]
        print('epoch = ', epoch, ' loss = {:.7}'.format(loss), \
              ' loss reduction = {:.7}'.format(loss_reduction), \
              ' correctly classified = {:.4%}'.format(accuracy))

        # You can apply your stop rule here if you would like

    return w1, b1, w2, b2


w1, b1, w2, b2 = nnet(x, y, alpha, num_epochs,num_train)

new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0


