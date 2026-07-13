# Fast Simulation of Volterra Processes using Random Fourier Features

This repository contains the code associated with the paper:

**Fast simulation of Volterra processes using random Fourier features with application to the log-stationary fractional Brownian motion**

Othmane Zarhali, Nicolas Langrené (2026)

arXiv: https://arxiv.org/abs/2603.02946

The repository provides numerical methods for the efficient simulation of stochastic Volterra processes based on a **Random Fourier Features (RFF)** approximation of the Volterra kernel. The proposed framework allows fast simulation while preserving the covariance structure of the underlying Gaussian processes.

The method is applied to the simulation of the **log-stationary fractional Brownian motion (log S-fBM)** model, a stochastic volatility model combining rough volatility and multifractal properties. :contentReference[oaicite:1]{index=1}

---

# Overview

A stochastic Volterra process is defined as

$$
X_t = \int_0^t K(t,s)dW_s,
$$

where:

- $W_t$ is a Brownian motion,
- $K(t,s)$ is a Volterra kernel controlling the memory structure.

Direct simulation of such processes can be computationally expensive due to the dense covariance structure induced by the kernel.

This repository implements an accelerated approach based on a spectral approximation:

$$
K(t,s) \approx K_N(t,s),
$$

where the kernel is represented using a finite collection of random Fourier features.

The resulting approximation reduces the computational complexity while maintaining the main statistical properties of the process.

---

# Main Features

The repository includes:

- Spectral density implementation
- Spectral Monte Carlo simulation using Hamiltonian Monte Carlo
- Random Fourier Features approximation of Volterra kernels
- Numerical experiments reproducing the results of the paper

---

# Repository Structure
