# WGAN 🌀

![WGAN Example](./images/fractal_example.png)

WGAN is a custom implementation of a Wasserstein Generative Adversarial Network (WGAN) for generating handwriter images. The project uses a unique approach to build the generator and discriminator models without relying on any deep learning frameworks. It serves as an educational resource for understanding the inner workings of GANs and their training process.

## Table of Contents
- [Introduction](#introduction)
- [Wasserstein GAN](#wasserstein-gan)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Generative Adversarial Networks (GANs) are a class of machine learning models that consist of two neural networks: a generator and a discriminator. The generator creates synthetic data samples, while the discriminator evaluates their authenticity. The generator and discriminator are trained in a two-player minimax game, where the generator aims to produce realistic samples and the discriminator tries to distinguish between real and fake samples.

## Wasserstein GAN

Wasserstein GAN (WGAN) is an improvement over the original GAN model, addressing issues such as training instability and mode collapse. The main difference between WGAN and GAN lies in the loss functions used for training. WGAN uses the Wasserstein distance, also known as the Earth Mover's distance, instead of the traditional Kullback-Leibler (KL) divergence used in GANs. This results in a more stable training process and better-quality generated samples.

![Wasserstein Loss](./images/wasserstein_loss.png)

## Installation

To set up the project, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/WGAN.git
cd WGAN