import numpy as np

import pickle as pickle


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict


def lsun_generator(filename, batch_size, data_dir):
    data = unpickle(data_dir + '/' + filename)

    images = data
    labels = np.zeros(len(images))

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        lsun_generator('celeba_batch', batch_size, data_dir)
    )
