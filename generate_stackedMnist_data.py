import os
import numpy as np
np.random.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.examples.tutorials.mnist import input_data


def generate_dict(data):
    dict_train = [[] for x in range(10)]
    for i in range(len(data.images)):
        k = np.argmax(data.labels[i])
        dict_train[k].append(i)
    return dict_train


def get_image(digit, dictlist, data):
    idx = np.random.randint(len(dictlist[digit]))
    pos = dictlist[digit][idx]
    return data.images[pos]


def assign_im(stacked_images, stacked_labels, data, dictlist, count, number):
    # Labels
    stacked_labels[count, number] = 1
    # Images
    digit_1 = number // 100
    digit_2 = (number // 10) % 10
    digit_3 = number % 10
    im_1 = get_image(digit_1, dictlist, data)
    im_2 = get_image(digit_2, dictlist, data)
    im_3 = get_image(digit_3, dictlist, data)
    im = np.concatenate((im_1, im_2, im_3), axis=0)
    stacked_images[count, :] = im
    return stacked_images, stacked_labels


def generate_data_1k(data, dictlist, _dsize, filename):
    stacked_images = np.zeros((_dsize, 3 * 28 * 28))
    stacked_labels = np.zeros((_dsize, 1000))
    prob_all = 0.3
    # generate all numbers
    _size_round_1 = int(_dsize * prob_all / 1000)  # all numbers
    if _size_round_1 == 0:
        _size_round_1 = 1
    count_1 = 0
    for iters in range(_size_round_1):
        for number in range(1000):
            stacked_images, stacked_labels = assign_im(stacked_images, stacked_labels, data, dictlist, count_1, number)
            count_1 = count_1 + 1

    # randomly generate some numbers => minor modes
    _size_round_2 = _dsize - _size_round_1 * 1000  # random minor
    dist = np.random.normal(700, 200, _size_round_2)
    count_2 = _size_round_1 * 1000
    for num in dist:
        if num < 0:
            number = 0
        elif num > 999:
            number = 999
        else:
            number = int(num)
        # Labels
        stacked_images, stacked_labels = assign_im(stacked_images, stacked_labels, data, dictlist, count_2, number)
        count_2 = count_2 + 1

    # shuffle order
    row_ord = np.arange(_dsize)
    np.random.shuffle(row_ord)
    stacked_images = stacked_images[row_ord, :]
    stacked_labels = stacked_labels[row_ord, :]
    stacked_dt = {"images": stacked_images, "labels": stacked_labels}
    np.save(filename, stacked_dt)


def generate_stacked_mnist(data_dir):
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    dict_train = generate_dict(mnist.train)
    generate_data_1k(mnist.train, dict_train, 50000, data_dir + 'stacked_train.npy')
