# Wave-Equations-with-PINNs

This repository contains code for solving 2D wave equations using Physics-Informed Neural Networks (PINNs) and Fourier-based PINNs (FBPINNs). The implementation leverages JAX for efficient computations and includes various utilities for seismic simulations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Wave Equation Problem](#wave-equation-problem)
  - [Training PINNs](#training-pinns)
  - [Training FBPINNs](#training-fbpinns)
- [Results](#results)
  

## Overview

The project aims to solve the time-dependent 2D wave equation with heterogeneous velocity model using PINNs and FBPINNs. The wave equation is given by:

The wave equation with a source term is given by:

 
$$
\frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u = f(x, t)
$$

Where:
- \( u(x, t) \) is the wave field as a function of space \( x \) and time \( t \),
- \( c \) is the wave propagation speed,
- \( \nabla^2 \) is the Laplacian operator (spatial second derivative),
- \( f(x, t) \) is the source term, representing external forces or inputs.


## Installation

To install the required dependencies, run:

 bash
pip install -r requirements.txt

Usage
Wave Equation Problem
The wave equation problem is defined in the WaveEquation3D class. It includes methods for initializing parameters, computing the velocity model (c_fn), sampling constraints, and defining the loss function.

Training PINNs
To train a PINN model, use the following code:


from fbpinns.trainers import PINNTrainer

<code>
# Initialize constants and problem
c = Constants(
    run="test",
    domain=RectangularDomainND,
    domain_init_kwargs=dict(
        xmin=np.array([-1, -1, 0]),
        xmax=np.array([1, 1, 1]),
    ),
    problem=WaveEquation3D,
    problem_init_kwargs=dict(
        c0=1, c1=3, sd=0.02,
    ),
    decomposition=RectangularDecompositionND,
    decomposition_init_kwargs=dict(
        subdomain_xs=subdomain_xs,
        subdomain_ws=subdomain_ws,
        unnorm=(0., 1.),
    ),
    network=FourierFCN,
    network_init_kwargs=dict(
        layer_sizes=[3, 32, 16, 1],
    ),
    ns=((80, 80, 80),),
    n_test=(60, 60, 10),
    n_steps=120000,
    optimiser_kwargs=dict(learning_rate=1e-3),
    summary_freq=1000,
    test_freq=200,
    show_figures=True,
    clear_output=True,
)

# Train the PINN model
run = PINNTrainer(c)
all_params, loss_log, x_batch_test, model_fns = run.train()
np.save("loss_pinn_src.npy", loss_log) </code>
