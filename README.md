# GDPP
Improved Generator loss to reduce mode-collapse and improve the generated samples quality. We use Determinantal Point Process (DPP) kernel to model the diversity within true data and fake data. Then we complement the generator loss with our DPP-inspired loss to diversify the generated data, imitating the true data diversity.

GDPP: https://arxiv.org/abs/1812.00068

Supplementary Materials: https://drive.google.com/open?id=18HrOSz3vCcVx7rso80SdC991j0dh9Sta

By: 
Mohamed Elfeki (*University of Central Florida*): elfeki@cs.ucf.edu

Camille Couprie, Morgane Riviere & Mohamed Elhoseiny 
(*Facebook Artificial Intelligence Research*): {coupriec,mriviere,elhoseiny}@fb.com


### DPP Inspiration

Inspired by DPP, we model a batch diversity using a kernel L. Our loss encourages generator G to synthesize a batch S_B of a diversity L_SB similar to the real data diversity L_DB , by matching their eigenvalues and eigenvectors. Generation loss aims at generating similar data points to the real, and diversity loss aims at matching the diversity manifold structures.

<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/GDPP_Teaser.png"/>
</p>


### Approach

Given a generator G and feature extraction function φ(·), the diversity kernel is constructed as L = φ(·). By modeling the diversity of fake and real batches, our loss matches their kernels L_SB and L_DB to encourage synthesizing samples of similar diversity to true data. We use the last feature map of the discriminator in GAN or the encoder in VAE as the feature representation φ.
  
<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/GDPP_Approach.png"/>
</p>


### Synthetic Data Experiments
Scatter plots of the true data (green dots) and generated data (blue dots) from different GAN methods trained on mixtures of 2D Gaussians arranged in a ring (top) or a grid (bottom).

<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/synthetic_qualitative.png"/>
</p>


### GDPP is generic
Our loss is generic to any generative model. We performed an experiment where we incorporated GDPP loss with both VAE(left) and GAN(right) on Stacked MNIST data. GDPP-GAN converges faster than GDPP-VAE and generates sharper samples.

![alt-text-1](https://github.com/M-Elfeki/GDPP/blob/master/Figures/vae_dpp_mnist.png "Generative Adversarial Network with GDPP") ![alt-text-2](https://github.com/M-Elfeki/GDPP/blob/master/Figures/stacked_mnist_qualitative.png "Variational AutoEncoder with GDPP")


### GDPP is scalable
Our loss can be applied to different architectures and data of various complexity. We embedded our loss with GAN trained on CIFAR-10(top) and CelebA(bottom).


<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/cifar_qualitative.jpg"/>
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/celeba_GDPPtrue_s5_iter_200000_avg.jpg"/>
</p>


### GDPP is immune to poor initialization

Since the weights of the generator are being initialized using a random number generator N (0, 1), the result of a generative model may be affected by poor initializations. In Figure 2 we show qualitative examples on 2D Grid data, where we use high standard deviation for the random number generator (i.e., σ > 100) as an example of poor initializations. Evidently, GDPP-GAN attains the true-data structure manifold even with poor initializations. On the other extreme, WGAN-GP tends to map the input noise to a disperse distribution covering all modes but with low-quality generations.

<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/poor_initialization_modified.jpg"/>
</p>



# Prerequisites
* Python 2.7, Tensorflow-gpu==1.0.0, Pytorch, SciPy, NumPy, Matplotlib, tqdm
* Cuda-8.0
* tflib==> attached with the code with minor modifications: https://github.com/nng555/tflib

# Models
Configuration for all models is specified in a list of constants at the top of the file. First, we have an implementation of the loss in both Tensorflow and Pytorch. The following file compute the GDPP loss of random feature and compute the difference based on the library implementation which is within a negligible margin.
* python GDPP_Loss_Tensorflow_Pytorch.py

The following file construct Stacked-MNIST dataset from the standard MNIST dataset.
* python generate_stackedMNIST_data.py

The following files represent experiments on synthetic data: synthetic 2D Ring and Grid, Stacked-MNIST, CIFAR-10. Every experiment is independent from the rest.
* python gdppgan_Synthetic.py
* python gdppgan_stackedMNIST.py
* python gdppgan_cifar.py


GDPP with Progressive Growing GANs (CelebA experiments) can be found here: https://github.com/facebookresearch/pytorch_GAN_zoo


