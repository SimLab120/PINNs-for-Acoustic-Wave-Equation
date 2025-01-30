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
- [Contributing](#contributing)
- [License](#license)

## Overview

The project aims to solve the time-dependent 2D wave equation with constant variable using PINNs and FBPINNs. The wave equation is given by:

\[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} - \frac{1}{c^2} \frac{\partial^2 u}{\partial t^2} = s(x, y, t) \]

where \( u \) is the wave field, \( c \) is the velocity, and \( s(x, y, t) \) is the source term.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
