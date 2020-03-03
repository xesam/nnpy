import numpy as np
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

train_file_path = 'D:/books-code/makeyourownneuralnetwork/mnist_dataset/mnist_train_100.csv'
test_file_path = 'D:/books-code/makeyourownneuralnetwork/mnist_dataset/mnist_test_10.csv'

with open(train_file_path) as train_file:
    train_list = train_file.readlines()

with open(test_file_path) as test_file:
    test_list = test_file.readlines()


def show_img(item):
    all_values = item.split(',')
    label = all_values[0]
    img_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(img_array, cmap='Greys', interpolation='None')
    print(label)
    plt.show()


n = NeuralNetwork(784, 100, 10, 0.3)


def train(network, item):
    all_values = item.split(',')
    label = int(all_values[0])
    targets = np.zeros(10) + 0.01
    targets[label] = 0.99
    img_array = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    network.train(img_array, targets)


def test(network, item):
    all_values = item.split(',')
    label = int(all_values[0])
    img_array = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    out = network.query(img_array)
    return label, np.argmax(out)


for train_item in train_list:
    train(n, train_item)

scorecard = []

for test_item in test_list:
    test_label, test_answer = test(n, test_item)
    if test_label == test_answer:
        scorecard.append(1)
    else:
        scorecard.append(0)
    print(test_label, test_answer)

scorecard_array = np.asarray(scorecard)
print('performance=', scorecard_array.sum() / scorecard_array.size)
