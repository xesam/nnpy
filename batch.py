import numpy as np
import matplotlib.pyplot as plt
import time

from TwoLayerNet import TwoLayerNet
from mnist.mnist import load_mnist


def now():
    return time.asctime(time.localtime(time.time()))


(train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, one_hot_label=True)

train_num = 1
train_size = train_img.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(784, 100, 10)

for i in range(train_num):
    print('train start', now())
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = train_img[batch_mask]
    t_batch = train_label[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print('train finish', now())

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(train_img, train_label)
        test_acc = network.accuracy(test_img, test_label)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
