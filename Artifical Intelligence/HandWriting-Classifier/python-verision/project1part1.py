## Written by: Drew Levin



import numpy as np


def data_loader(file):
    #using numpy seperate numbers using delimiter
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
print('data loading done')

#classify numbers 9 and 3
test_labels = [9, 3]

#Get index where ys from training set are in 9 or 3
indices = np.where(np.isin(y_train, test_labels))[0]

x = x_train[indices]
y = y_train[indices]

# set 9's equal to 0 and 3's equal to 1
y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1

# Number of runs
num_epochs = 10
# Learning rate
alpha = 0.1

# Get dimension of x
m = x.shape[1]

# generate weights
w = np.random.rand(m)

# generate bias
b = np.random.rand()

# pick random loss(big number)
loss_previous = 100000

# loop for number of runs
for epoch in range(num_epochs):
    # matrix multiplication with x's and weights then add bias
    a = x @ w + b

    a = 1 / (1 + np.exp(-a))
    # bound items in a to avoid log(0):
    a = np.clip(a, .001, .999)

    w -= alpha * (x.T) @ (a - y)

    b -= alpha * (a - y).sum()

    loss = - np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    loss_reduction = loss_previous - loss
    loss_previous = loss

    accuracy = sum((a > 0.5).astype(int) == y) / len(y)

    print('epoch = ', epoch, ' loss = {:.7}'.format(loss), \
          ' loss reduction = {:.7}'.format(loss_reduction), \
          ' correctly classified = {:.4%}'.format(accuracy))

# New Test
for i in w:
    print("{:.4f}".format(i), end=",")
print(b)

test = open("test.txt", "r")
test_array = []
for line in test:
    test_array.append(list(map(lambda x: int(x) / 255, line.split(','))))
tests_final = np.array(test_array)

final_output = 1 / (1 + np.exp(-(np.matmul(w, np.transpose(tests_final)) + b)))
print(",".join("{:.2f}".format(i) for i in final_output))
print(",".join(map(lambda x: str(int(x)), final_output)))