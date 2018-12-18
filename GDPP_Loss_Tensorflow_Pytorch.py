import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import torch
import torch.nn.functional as f
import tensorflow as tf

# Implementing the GDPP loss in both Tensorflow and Pytorch
np.random.seed(1)


def compute_gdpp(phi_fake, phi_real):
    def compute_diversity(phi):
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        eig_vals, eig_vecs = torch.eig(S_B, eigenvectors=True)
        return eig_vals[:, 0], eig_vecs

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v) / (max_v - min_v)

    fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
    real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
    # Scaling factor to make the two losses operating in comparable ranges.
    magnitude_loss = 0.0001 * f.mse_loss(target=real_eig_vals, input=fake_eig_vals)
    structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
    normalized_real_eig_vals = normalize_min_max(real_eig_vals)
    weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss))
    return magnitude_loss + weighted_structure_loss


def compute_diversity_loss(phi_fake, phi_real):
    def compute_diversity(phi):
        phi = tf.nn.l2_normalize(phi, 1)
        Ly = tf.tensordot(phi, tf.transpose(phi), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(phi_fake)
    real_eig_val, real_eig_vec = compute_diversity(phi_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return tf.cast(eigen_values_loss, tf.float64) + weighted_eigen_vectors_loss


# Sanity Check: Compare the two implementations (Tensorflow vs. Pytorch)
losses = []
session = tf.InteractiveSession()
for i in range(100):
    phi_1 = np.random.random((512, 128)) * 1000000.0
    phi_2 = np.random.random((512, 128)) * 1000000.0
    q1 = session.run(compute_diversity_loss(tf.constant(phi_1), tf.constant(phi_2)))
    q2 = compute_gdpp(torch.tensor(phi_1), torch.tensor(phi_2)).item()

    losses.append(np.abs(q1 - q2))

losses = np.array(losses)
print losses.mean(), losses.std()

# Slightly different results based on the eigen-values.
# Tensorflow and Pytorch have different methods to do the eigen-decomposition,
# The results differ in +/- 0.001482 on average with a std of 0.0011018
