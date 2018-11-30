from tqdm import tqdm
import numpy as np, itertools, collections, os, random, math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

# Change the write Dir to where you want to save the models & Plots
write_dir = '/home/elfeki/Workspace/GDPP-GAN/Synthetic/'

if not os.path.exists(write_dir + 'models'):
    os.makedirs(write_dir + 'models')
if not os.path.exists(write_dir + 'Plots'):
    os.makedirs(write_dir + 'Plots')

params = dict(
    x_dim=2,
    z_dim=256,
    beta1=0.5,
    epsilon=1e-8,
    viz_every=500,
    batch_size=512,
    max_iter=25000,
    gen_learning_rate=1e-3,
    disc_learning_rate=1e-4,
    eig_vals_loss_weight=0.0001,
    number_evaluation_samples=2500,
)

ds = tf.contrib.distributions
slim = tf.contrib.slim


def sample_ring(batch_size, n_mixture=8, std=0.01, radius=1.0):
    """Gnerate 2D Ring"""
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size, seed=random_seed)


def sample_grid(batch_size, num_components=25, std=0.05):
    """Generate 2D Grid"""
    cat = ds.Categorical(tf.zeros(num_components, dtype=tf.float32))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)
    sigmas = [np.array([std, std]).astype(np.float32) for i in range(num_components)]
    components = list((ds.MultivariateNormalDiag(mu, sigma)
                       for (mu, sigma) in zip(mus, sigmas)))
    data = ds.Mixture(cat, components)
    return data.sample(batch_size, seed=random_seed)


def generator(z, output_dim=params['x_dim'], n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer,
                       activation_fn=tf.nn.tanh)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x


def discriminator(x, n_hidden=128, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h_1 = slim.fully_connected(x, n_hidden, activation_fn=tf.nn.tanh)
        h_2 = slim.fully_connected(h_1, n_hidden, activation_fn=None)
        h = tf.nn.tanh(h_2)
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d, h


def evaluate_samples(generated_samples, data, model_num, is_ring_distribution=True):
    generated_samples = generated_samples[:params['number_evaluation_samples']]
    data = data[:params['number_evaluation_samples']]

    if is_ring_distribution:
        thetas = np.linspace(0, 2 * np.pi, 8)
        xs, ys = np.sin(thetas), np.cos(thetas)
        MEANS = np.stack([xs, ys]).transpose()
        std = 0.01
    else:  # Grid Distribution
        MEANS = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                         range(-4, 5, 2))], dtype=np.float32)
        std = 0.05

    l2_store = []
    for x_ in generated_samples:
        l2_store.append([np.sum((x_ - i) ** 2) for i in MEANS])

    mode = np.argmin(l2_store, 1).flatten().tolist()
    dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
    mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i]) <= (3 * std)]

    sns.set(font_scale=2)
    f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(generated_samples[:, 0], generated_samples[:, 1], cmap=cmap, ax=ax1, n_levels=100, shade=True,
                clip=[[-6, 6]] * 2)
    sns.kdeplot(data[:, 0], data[:, 1], cmap=cmap, ax=ax2, n_levels=100, shade=True, clip=[[-6, 6]] * 2)

    plt.figure(figsize=(5, 5))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], edgecolor='none')
    plt.scatter(data[:, 0], data[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.savefig(write_dir + 'Plots/GDPPGAN_Final_{}.png'.format(model_num))
    plt.clf()

    high_quality_ratio = np.sum(collections.Counter(mode_counter).values()) / float(params['number_evaluation_samples'])
    print('Model: %d || Number of Modes Captured: %d' % (model_num, len(collections.Counter(mode_counter))))
    print('Percentage of Points Falling Within 3 std. of the Nearest Mode %f' % high_quality_ratio)


def compute_diversity_loss(h_fake, h_real):
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
    eigen_values_loss = params['eig_vals_loss_weight'] * \
                        tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss


def run_gan(model_num):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)

        if model_num % 2 == 0:
            data = sample_ring(params['batch_size'])
        else:
            data = sample_grid(params['batch_size'])
        noise = ds.Normal(tf.zeros(params['z_dim']), tf.ones(params['z_dim'])).sample(params['batch_size'],
                                                                                      seed=random_seed)

        # Construct generator and discriminator nets
        with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
            samples = generator(noise, output_dim=params['x_dim'])
            real_score, real_h = discriminator(data)
            fake_score, fake_h = discriminator(samples, reuse=True)

        # Objectives
        disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(real_score)))
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.ones_like(real_score)))

        # Adding GDPP-Loss
        diversity_loss = compute_diversity_loss(fake_h, real_h)
        gen_loss = 0.5 * (gen_loss + diversity_loss)

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

        # Discriminator & Generator update
        d_opt_train = tf.train.AdamOptimizer(params['disc_learning_rate'], beta1=params['beta1'],
                                             epsilon=params['epsilon'])
        d_train_op = d_opt_train.minimize(disc_loss, var_list=disc_vars)
        g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'],
                                             epsilon=params['epsilon'])
        g_train_op = g_train_opt.minimize(gen_loss, var_list=gen_vars)

        # Train !!
        sess = tf.InteractiveSession(config=run_config)
        sess.run(tf.global_variables_initializer())

        for i in tqdm(xrange(params['max_iter'])):
            _, _ = sess.run([g_train_op, d_train_op])

            if i % params['viz_every'] == 0:
                xx, yy = sess.run([samples, data])
                plt.figure(figsize=(5, 5))
                plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
                plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(write_dir + 'Plots/DPPGAN_{}.png'.format(model_num))
                plt.clf()
                plt.close()

        range_num_samples = range(int(math.ceil(params['number_evaluation_samples'] / float(params['batch_size']))))
        xx = np.vstack([sess.run(samples) for _ in range_num_samples])
        yy = np.vstack([sess.run(data) for _ in range_num_samples])
        evaluate_samples(xx, yy, model_num, is_ring_distribution=(model_num % 2 == 0))
        saver = tf.train.Saver()
        saver.save(sess, write_dir + "models/model_{}.ckpt".format(model_num))


if __name__ == '__main__':
    # Specify a model id to experiment with.
    # Even numbers for 2D Ring, Odd numbers for 2D Grid
    run_gan(model_num=0)  # 2D Ring
    run_gan(model_num=1)  # 2D Grid
