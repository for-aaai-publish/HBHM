# HBHM: Hierarchical Bayesian Hyperbolic Model

## Overview

This is the implementation of HBHM (Hierarchical Bayesian Hyperbolic Model), a geometry-aware hyperbolic prior framework for hierarchical uncertainty quantification. This project provides a principled approach to combining geometric fidelity with probabilistic reasoning in hierarchical classification.

## Core Contributions

- **Geometry-Aware Hyperbolic Prior**: Translates the topology of label hierarchy into the statistical parameters of a Bayesian prior
- **Dual-pathway Regularization**: Concurrently shapes the prior's geometric foundation and supervises the posterior's representation space
- **Modular Decoupled Strategy**: Enables pre-trained DNNs to be efficiently transformed into BNNs with hierarchical uncertainty quantification capabilities

## Paper Figure Showcase

### Figure 1: Visualization of Learned Categorical Hierarchy and Data Distribution in Hyperbolic Space

![Poincaré Embedding Visualization](https://res.cloudinary.com/dyybhhdfg/image/upload/v1754404374/poincare_embedding_with_data_xjsxz0.png)

*Figure 1 shows class-level hierarchical nodes embedded in the Poincaré disk. t-SNE projections of ID (CIFAR-10) and OOD (SVHN) data demonstrate the model's separation of data types. The size of the shaded areas represents learned uncertainty. Subclass nodes' distance from the origin are determined by the learned embedding vectors, reflecting the model's comprehension of hierarchical depth.*

### Figure 2: Architecture and Data Flow of Bayesian Head Training

![Architecture Diagram](https://res.cloudinary.com/dyybhhdfg/image/upload/v1754404429/Two_stage_symaut.png)

*Figure 2 shows the overall architecture of HBHM, including the pre-trained backbone network, hierarchy-aware Bayesian head, and the data flow of the dual-pathway regularization mechanism.*

## Project Structure

```
HBHM_Clone/
├── core/                           # Core code package
│   ├── training/                   # Training-related modules
│   │   ├── losses/                 # Loss function implementations
│   │   │   ├── vfe_loss.py        # Variational Free Energy loss
│   │   │   ├── prior_loss.py      # Prior geometry shaping loss
│   │   │   ├── posterior_loss.py  # Posterior structural regularization loss
│   │   │   ├── kl_loss.py         # KL divergence loss
│   │   │   └── modular_loss.py    # Modular loss combination
│   │   ├── trainer.py             # Main trainer
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── optimizers.py          # Hyperbolic geometry optimizers
│   ├── models/                    # Model definitions
│   │   ├── hbhm_model.py         # HBHM main model
│   │   └── backbone.py           # Backbone network
│   ├── vi/                       # Variational inference modules
│   │   ├── low_rank_diag_vi.py   # Low-rank plus diagonal VI
│   │   └── base_vi_layer.py      # Variational inference base class
│   ├── priors/                   # Prior distribution modules
│   │   ├── structured_direct_prior.py  # Structured direct prior (HPG)
│   │   └── base_prior.py         # Prior base class
│   ├── utils/                    # Utility functions
│   │   ├── hierarchy_utils.py    # Hierarchy structure processing
│   │   ├── linalg_utils.py       # Hyperbolic geometry linear algebra
│   │   └── kl_calculator.py      # KL divergence calculation
│   └── data/                     # Data processing
│       ├── hierarchy.py          # Hierarchy structure data
│       ├── cifar10.py           # CIFAR-10 data loading
│       └── cifar10_hierarchy.json # CIFAR-10 hierarchy definition
├── examples/                     # Usage examples
│   ├── train_cifar10.py         # CIFAR-10 training example
├── configs/                      # Configuration files
│   ├── default.yaml             # Default configuration
│   └── cifar10.yaml             # CIFAR-10 configuration
├── checkpoints/                 # Model checkpoints
├── weights/                     # Pre-trained weights
├── dataset/                        # Datasets
└── logs/                        # Training logs
```

## Quick Start

### Environment Setup

Create environment using conda:

```bash
conda env create -f environment.yml
conda activate HBHM
```


## Key Features
- Freezes pre-trained backbone network as feature extractor
- Only trains hierarchy-aware Bayesian head
- Efficiently "Bayesianizes" deterministic models

Code Implementation Coming Soon

## License
This project is licensed under the MIT License
