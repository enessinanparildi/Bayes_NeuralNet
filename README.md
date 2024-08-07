# MCMC Sampling from Bayesian Neural Network
Importance sampling, metropolis-hastings, random walk metropolis to sample from a bayesian neural network 

This project implements a neural network with Bayesian inference capabilities. It includes various sampling methods for posterior estimation.

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Classes](#classes)
4. [Main Features](#main-features)
5. [Usage](#usage)
6. [Sampling Methods](#sampling-methods)

## Overview

This project implements a feed-forward neural network with three layers. It uses Bayesian inference techniques to estimate the posterior distribution of the network parameters. Various sampling methods are implemented to perform this inference.

## Dependencies

- numpy
- scipy
- joblib
- preprocessing (custom module)
- utils (custom module)
- samplers (custom module)

## Classes

### Network

The main class that implements the neural network and related computations.

Key methods:
- `__init__`: Initializes the network with layer information and data.
- `forward_neural_network_calculation`: Performs a forward pass through the network.
- `likehood_sample`: Computes the likelihood of a sample.
- `log_prior_sample`: Computes the log prior of a sample.
- `get_unnormalized_weight_posterior`: Computes the unnormalized posterior of the weights.

## Main Features

1. Three-layer feed-forward neural network
2. Bayesian inference for network parameters
3. Various sampling methods for posterior estimation
4. Parallel processing capabilities for improved performance

## Usage

The main function demonstrates how to use the Network class and various samplers:

1. Load or generate dataset
2. Create a Network instance
3. Choose and run a sampling method
4. Analyze results

Example:

```python
bayes_net = Network((2, 5, 2), X_train, X_test, y_train)
rwm_sampler = samplers.RandomWalkMetropolis(stdval=stdval_rwm, num_of_run=num_of_run_rwm, network=bayes_net, name=name)
rwm_sampler.main_loop()
print('acceptance_rate_rwm = ' + str(rwm_sampler.get_acceptence_rate()))
