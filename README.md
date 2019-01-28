# GDPP
Improved Generator loss to reduce mode-collapse and improve the generated samples quality. We use Determinantal Point Process (DPP) kernel to model the diversity within true data and fake data. Then we complement the generator loss with our DPP-inspired loss to diversify the generated data, imitating the true data diversity.

GDPP: https://arxiv.org/abs/1812.00068

Supplementary Materials: https://drive.google.com/open?id=18HrOSz3vCcVx7rso80SdC991j0dh9Sta

By: 
Mohamed Elfeki (*University of Central Florida*): elfeki@cs.ucf.edu

Camille Couprie, Morgane Riviere & Mohamed Elhoseiny 
(*Facebook Artificial Intelligence Research*): {coupriec,mriviere,elhoseiny}@fb.com

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


The experiments in the Progressive Growing GANs can be done by embedding the GDPP loss within the official implementation in: https://github.com/tkarras/progressive_growing_of_gans


# To-DO List
Upload the CelebA experiments using Progressive-Growing GANs.


# Figures

<p align="center">
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_1.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_2.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_3.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_4.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_5.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_6_2.png"/>
  
  
  <img src ="https://github.com/M-Elfeki/GDPP/blob/master/Figures/Fig_7.png"/>
  
</p>
