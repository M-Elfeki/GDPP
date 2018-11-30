import math, os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tqdm import tqdm

np.random.seed(1)
import tensorflow as tf

tf.set_random_seed(1)

import tflib as lib
from tflib.linear import Linear
from tflib.batchnorm import Batchnorm
from tflib.deconv2d import Deconv2D
from tflib.conv2d import Conv2D

from scipy.stats import entropy


class DPPGAN(object):
    def __init__(self, model_path):
        self.dim_h = 64
        self.dim_z = 128
        self.im_size = 28
        self.batch_size = 64
        self.dim_x = 3 * self.im_size * self.im_size
        self.model_dir = create_dir(model_path + '/models/')

        # Noise Vector & Input Images
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim_z])
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x])

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            fc1 = Linear('Generator.Input', self.dim_z, 4 * 4 * 4 * self.dim_h, z)
            fc1 = Batchnorm('Generator.BN1', [0], fc1)
            fc1 = tf.nn.relu(fc1)
            out_fc1 = tf.reshape(fc1, [-1, 4 * self.dim_h, 4, 4])

            deconv1 = Deconv2D('Generator.2', 4 * self.dim_h, 2 * self.dim_h, 5, out_fc1)
            deconv1 = Batchnorm('Generator.BN2', [0, 2, 3], deconv1)
            deconv1 = tf.nn.relu(deconv1)
            out_deconv1 = deconv1[:, :, :7, :7]

            deconv2 = Deconv2D('Generator.3', 2 * self.dim_h, self.dim_h, 5, out_deconv1)
            deconv2 = Batchnorm('Generator.BN3', [0, 2, 3], deconv2)
            out_deconv2 = tf.nn.relu(deconv2)

            deconv3 = Deconv2D('Generator.5', self.dim_h, 3, 5, out_deconv2)
            out_deconv3 = tf.sigmoid(deconv3)
            return tf.reshape(out_deconv3, [-1, self.dim_x])

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            im = tf.reshape(x, [-1, 3, self.im_size, self.im_size])

            conv1 = Conv2D('Discriminator.1', 3, self.dim_h, 5, im, stride=2)
            out_conv1 = LeakyReLU(conv1)

            conv2 = Conv2D('Discriminator.2', self.dim_h, 2 * self.dim_h, 5, out_conv1, stride=2)
            out_conv2 = LeakyReLU(conv2)

            conv3 = Conv2D('Discriminator.3', 2 * self.dim_h, 4 * self.dim_h, 5, out_conv2, stride=2)
            out_conv3 = LeakyReLU(conv3)

            fc = tf.reshape(out_conv3, [-1, 4 * 4 * 4 * self.dim_h])
            out_fc = Linear('Discriminator.Output', 4 * 4 * 4 * self.dim_h, 1, fc)
            return tf.reshape(out_fc, [-1]), fc

    def sample_z(self, m, n):
        # sample from a gaussian distribution
        return np.random.normal(size=[m, n], loc=0, scale=1)

    def compute_diversity_loss(self, h_fake, h_real):  # GDPP Loss
        def compute_diversity(h):
            h = tf.nn.l2_normalize(h, 1)
            Ly = tf.tensordot(h, tf.transpose(h), 1)
            eig_val, eig_vec = tf.self_adjoint_eig(Ly)
            return eig_val, eig_vec

        def normalize_min_max(eig_val):
            # Min-max-Normalize Eig-Values
            return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                          tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))

        fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
        real_eig_val, real_eig_vec = compute_diversity(h_real)
        # Used a weighing factor to make the two losses operating in comparable ranges.
        eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
        eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
        normalized_real_eig_val = normalize_min_max(real_eig_val)
        weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
        return eigen_values_loss + weighted_eigen_vectors_loss

    def train(self, gen, n_iters=15000):
        batch_size = self.batch_size
        zin = tf.get_variable("g_z", [self.batch_size, self.dim_z], initializer=tf.random_uniform_initializer(-1, 1))
        zsig = tf.get_variable("g_sig", [self.batch_size, self.dim_z], initializer=tf.constant_initializer(0.2))
        inp = tf.add(zin, tf.multiply(self.Z, zsig))
        G_sample = self.generator(inp)
        images = tf.reshape(self.X, [self.batch_size, 28, 28, 3])
        D_fake, h_fake = self.discriminator(G_sample)
        D_real, h_real = self.discriminator(images, reuse=True)

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real) * 0.9))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_real) * 0.1))
        D_loss = D_loss_fake + D_loss_real

        G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_real) * 0.9))

        diversity_loss = self.compute_diversity_loss(h_fake, h_real)
        G_loss = 0.5 * (G_loss + diversity_loss)

        disc_params = lib.params_with_name('Discriminator')
        gen_params = lib.params_with_name('Generator')

        D_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(D_loss, var_list=disc_params)
        G_solver = tf.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(G_loss, var_list=gen_params)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            tf.set_random_seed(1)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for it in tqdm(xrange(n_iters)):
                _data, _ = next(gen)
                _ = sess.run(D_solver, feed_dict={self.X: _data, self.Z: self.sample_z(batch_size, self.dim_z)})
                _ = sess.run(G_solver, feed_dict={self.X: _data, self.Z: self.sample_z(batch_size, self.dim_z)})
                if np.mod(it, 2000) == 2:
                    saver.save(sess, self.model_dir + 'dppgan_mnist.ckpt', global_step=it)

            saver.save(sess, self.model_dir + 'dppgan_mnist_final.ckpt')

    def generate_sample(self, m=26000):
        sess = tf.Session()
        tf.set_random_seed(1)
        G_sample = self.generator(self.Z, reuse=True)
        saver = tf.train.Saver()
        saver.restore(sess, self.model_dir + 'dppgan_mnist_final.ckpt')
        samples = []
        for _ in range(int(math.ceil(m / self.batch_size)) + 1):
            samples.extend(sess.run(G_sample, feed_dict={self.Z: self.sample_z(self.batch_size, self.dim_z)}))
        sess.close()
        if len(np.array(samples).shape) == 2:
            samples = np.array(samples[:m]).reshape(m, 3, self.im_size ** 2)
        else:
            samples = np.array(samples[:m]).transpose(0, 3, 1, 2).reshape(-1, 3, self.im_size ** 2)
        return samples


####################################################################################
# ------------------------------ Evaluation Network ------------------------------ #
####################################################################################
class MNIST(object):
    def __init__(self, model_name):
        # Create the model
        self.model_name = model_name
        self.x = tf.placeholder(tf.float32, [None, 784])
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def deepnn(self, x):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        return y_conv

    def build_model(self, mnist):
        # Build the graph for the deep net
        y_conv = self.deepnn(self.x)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        params = tf.trainable_variables()
        train_step = tf.train.AdamOptimizer(
            1e-4).minimize(cross_entropy, var_list=params)

        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in tqdm(xrange(20000)):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                train_step.run(
                    feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            saver.save(sess, self.model_name)

    def classify(self, inputs):
        y_conv = self.deepnn(self.x)
        sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(sess, self.model_name)
        # from IPython import embed; embed()
        y_pred = sess.run(y_conv, feed_dict={
            self.x: inputs, self.keep_prob: 1.0})
        y = np.argmax(y_pred, 1)
        return y


##################################################################################
# ------------------------------ Helper Functions ------------------------------ #
##################################################################################

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_dist(p, n):
    pk = np.zeros(n)
    for x in p:
        pk[x] += 1
    for i in range(n):
        pk[i] = pk[i] * 1.0 / len(p)
    return pk


def inf_train_gen(dataset_path, batch_size=64):
    ds = np.load(dataset_path).item()
    images, labels = ds['images'], ds['labels']
    ds_size = labels.shape[0]
    while True:
        for i in range(int(ds_size / batch_size)):
            start = i * batch_size
            end = (i + 1) * batch_size
            yield images[start:end], labels[start:end]


###############################################################################
# ------------------------------ Main Function ------------------------------ #
###############################################################################
if __name__ == '__main__':
    main_dir = '/home/elfeki/Workspace/GDPP-GAN/MNIST/'
    data_dir = main_dir + '/data/'

    # # To generate the Stacked-MNIST data- One time
    # from generate_stackedMnist_data import generate_stacked_mnist
    # create_dir(data_dir)
    # generate_stacked_mnist(data_dir)

    dppgan = DPPGAN(main_dir)
    gen = inf_train_gen(data_dir + 'stacked_train.npy')
    dppgan.train(gen)
    samples = dppgan.generate_sample()

    np.random.seed(1)
    tf.reset_default_graph()
    tf.set_random_seed(1)
    nnet = MNIST(main_dir)

    # # To train the CNN-classifier for evaluating the model- One time
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets(data_dir, one_hot=True)
    # nnet.build_model(mnist)

    digit_1 = nnet.classify(samples[:, 0])
    digit_2 = nnet.classify(samples[:, 1])
    digit_3 = nnet.classify(samples[:, 2])
    y_pred = []
    for i in range(len(digit_1)):
        y_pred.append(digit_1[i] * 100 + digit_2[i] * 10 + digit_3[i])
    x = np.unique(y_pred)

    ds = np.load(data_dir + 'stacked_train.npy').item()
    labels = np.array(ds['labels'])
    y_true = [np.argmax(lb) for lb in labels]
    qk = get_dist(y_true, 1000)
    pk = get_dist(y_pred, 1000)
    kl_score = entropy(pk, qk)
    print("#Modes: %d, KL-score: %.3f\n\n" % (len(x), kl_score))
