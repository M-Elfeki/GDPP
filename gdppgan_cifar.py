import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tqdm import tqdm

np.random.seed(1)
import tensorflow as tf

import tflib as lib
from tflib.linear import Linear
from tflib.batchnorm import Batchnorm
from tflib.deconv2d import Deconv2D
from tflib.conv2d import Conv2D
from tflib.inception_score import get_inception_score
from tflib.cifar10 import load
from tflib.save_images import save_images

MAIN_DIR = '/home/elfeki/Workspace/GDPP-GAN/CIFAR/'
DATA_DIR = MAIN_DIR + 'data/'
# Download CIFAR-10 (Python version) from
# https://www.cs.toronto.edu/~kriz/cifar.html
# and put them in DATA_DIR.
if not os.path.exists(MAIN_DIR + 'models/'):
    os.makedirs(MAIN_DIR + 'models/')

ITERS = 100000
BATCH_SIZE = 64
DIM = 128  # This overfits substantially; you're probably better off with 64
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4 * 4 * 4 * DIM, noise)
    output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4 * DIM, 4, 4])

    output = Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5, output)
    output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
    output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = Deconv2D('Generator.5', DIM, 3, 5, output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.2', DIM, 2 * DIM, 5, output, stride=2)
    output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5, output, stride=2)
    output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    hidden_features = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])
    output = Linear('Discriminator.Output', 4 * 4 * 4 * DIM, 1, hidden_features)
    return tf.reshape(output, [-1]), hidden_features


def compute_diversity_loss(h_fake, h_real):  # GDPP-Loss
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = tf.tensordot(h, tf.transpose(h), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss


def cifar_get_inception_score(session, samples_100):
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return get_inception_score(list(all_samples))


def inf_train_gen(batch_size=BATCH_SIZE):
    # Dataset iterators
    train_gen, dev_gen = load(batch_size, data_dir=DATA_DIR)
    while True:
        for images, _ in train_gen():
            yield images


def train_network():
    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_data = 2 * ((tf.cast(real_data_int, tf.float32) / 255.) - .5)
    fake_data = Generator(BATCH_SIZE)

    disc_real, h_real = Discriminator(real_data)
    disc_fake, h_fake = Discriminator(fake_data)
    diverstiy_cost = compute_diversity_loss(h_fake, h_real)

    # Standard GAN Loss
    gen_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.9))
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.1))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real) * 0.9))
    disc_cost /= 2.

    # GDPP Penalty
    gen_cost += diverstiy_cost
    gen_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                  var_list=lib.params_with_name(
                                                                                      'Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                   var_list=lib.params_with_name(
                                                                                       'Discriminator.'))

    # Train loop
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        gen = inf_train_gen()
        for iteration in tqdm(xrange(ITERS)):
            _data = gen.next()
            if iteration > 0:
                session.run(gen_train_op, feed_dict={real_data_int: _data})
            session.run(disc_train_op, feed_dict={real_data_int: _data})

            if iteration > 0 and iteration % 1000 == 0:
                saver.save(session, MAIN_DIR + 'models/gdpp_gan.ckpt', global_step=iteration)
        saver.save(session, MAIN_DIR + 'models/gdpp_gan_final.ckpt')


def evaluate_network(n_samples):
    np.random.seed(1)
    tf.set_random_seed(1)
    samples = Generator(n_samples)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/gdpp_gan_final.ckpt')
    inception_score = cifar_get_inception_score(session, samples)
    print('Evaluation Inception Score: {:.2f} +/- {:.2f}'.format(inception_score[0], inception_score[1]))


def evaluate_inference_via_optimization(n_samples):
    np.random.seed(1)
    tf.set_random_seed(1)
    samples = Generator(n_samples)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/gdpp_gan_final.ckpt')

    gen = inf_train_gen()
    data = gen.next()
    data = 2 * ((data / 255.) - .5)

    sample_z_dim = 128
    z_var = tf.get_variable("z_var", shape=[n_samples, sample_z_dim], dtype=tf.float32, trainable=False)
    z_var_pl = tf.placeholder(dtype=tf.float32, shape=[n_samples, sample_z_dim], name="z_var_placeholder")
    z_var_assign = tf.assign(z_var, z_var_pl, name="z_var_assign")

    targets = data[np.random.randint(data.shape[0], size=n_samples)]

    g_loss = tf.nn.l2_loss(samples - targets)
    a = (samples + 1.) / 2.
    b = (targets + 1.) / 2.
    mse = ((a - b) ** 2)
    mse_2d = tf.reshape(mse, [n_samples, 32 * 32 * 3])
    mse = tf.reduce_mean(mse_2d, axis=1, keep_dims=True)

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=g_loss, var_list=[z_var], method='L-BFGS-B',
                                                       options={'maxiter': 25, 'disp': False})

    # run the optimization from 3 different initializations
    results_images = []
    results_errors = []
    num_of_random_restarts = 3

    for i in xrange(num_of_random_restarts):
        z_sample = np.random.normal(0, 1, size=(n_samples, sample_z_dim))
        session.run(z_var_assign, {z_var_pl: z_sample})
        optimizer.minimize(session)

        generated_samples = session.run(samples)
        generated_samples_mse = session.run(mse)
        results_images.append(generated_samples)
        results_errors.append(generated_samples_mse)

    # select the best out of all random restarts
    best_images = np.zeros_like(results_images[0])
    best_images_errors = np.zeros_like(results_errors[0])
    for image_index in xrange(n_samples):
        best_img = results_images[0][image_index]
        best_img_error = results_errors[0][image_index][0]
        for indep_run_index in xrange(1, num_of_random_restarts):
            if best_img_error > results_errors[indep_run_index][image_index][0]:
                best_img_error = results_errors[indep_run_index][image_index][0]
                best_img = results_images[indep_run_index][image_index]
        best_images[image_index] = best_img
        best_images_errors[image_index][0] = best_img_error

    best_images_errors_i = np.array(best_images_errors)
    best_images_errors = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(1):
        best_images_errors = np.vstack((best_images_errors, best_images_errors_i))

    print('Evaluation Inference Via Optimization: {:.5f} +/- {:.5f}'.format(np.mean(best_images_errors),
                                                                            np.std(best_images_errors)))


if __name__ == "__main__":
    train_network()
    evaluate_network(2000)
    evaluate_inference_via_optimization(2000)
