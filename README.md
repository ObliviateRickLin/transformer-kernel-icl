# transformer-kernel-icl

This repository unofficially implements and partially reproduces the experiments from the paper:

**"Transformers Implement Functional Gradient Descent to Learn Non-Linear Functions In Context"**  
Xiang Cheng, Yuxin Chen, Suvrit Sra  
[arXiv:2312.06528v6](https://arxiv.org/abs/2312.06528v6)

## Overview

This codebase provides:
- **Model implementation**: The core Transformer architecture as described in the paper, with flexible kernelized attention mechanisms (linear, ReLU, exponential, softmax, etc.), and support for functional gradient descent interpretation.
- **Experiment settings**: Scripts and configuration for running the main experiments (e.g., Figure 1) in the paper, including data generation, training, and evaluation.
- **Partial reproduction**: Selected experiments from the original work are implemented to verify theoretical claims and empirical results.

## Features

- Modular implementation of generalized attention layers supporting different kernel functions.
- Training and evaluation scripts for in-context learning tasks with synthetic data.
- Learning rate scheduling, gradient clipping, and other best practices for stable training.
- Unit tests for key model components.

## Reference

If you use this code or ideas from this repository, please cite the original paper:

> **Transformers Implement Functional Gradient Descent to Learn Non-Linear Functions In Context**  
> Xiang Cheng, Yuxin Chen, Suvrit Sra  
> [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) 