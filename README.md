# Deep Learning for Spatio-Temporal Model Estimation

This repository contains the work conducted during a Master's internship focused on estimating parameters of spatio-temporal models using advanced deep learning techniques. The main approach leverages Gaussian random fields modeled by Matérn covariance functions and explores the use of Graph Neural Networks (GNNs) for parameter estimation.

> **Note**: A detailed report in French is included in this repository. Please refer to it for comprehensive explanations and mathematical derivations.

## Project Overview

The aim of this internship was to develop an amortized inference framework for estimating parameters of spatial and spatio-temporal processes. By simulating datasets from Matérn Gaussian fields, a neural network was trained to instantly infer model parameters.

### Key Contributions
- **Simulation of Training Data**: Using Matérn Gaussian fields sampled through a double Poisson process.
- **Neural Network Architecture**: Incorporating Graph Neural Networks (GNNs) for irregular spatial data and convolutional methods for structured grids.
- **Amortized Inference**: Enabling near-instantaneous parameter estimation after training.
- **Implementation**: Reconstruction of a GNN-based model using PyTorch Geometric, with optimized graph convolution methods.

## Features

### Model Architecture
The implemented model includes:
1. **Graph Convolution**: For processing graph-based spatial data.
2. **Readout Module**: Summarizes graph features into a compact vector.
3. **Mapping Module**: A multi-layer perceptron (MLP) that maps readout vectors to the target parameter space.

### Data Representation
- Training samples are represented as graphs where nodes correspond to spatial locations.
- Graph edges connect nodes within a defined distance, and edge weights depend on pairwise distances.
- Use of **DeepSets** to aggregate multiple realizations of Gaussian fields sharing the same parameters.

### Optimization Techniques
The graph convolution implementation underwent multiple iterations, with a focus on efficiency:
- Early versions relied on nested loops for convolution.
- The final implementation leverages vectorized operations and PyTorch Geometric’s message-passing framework.

## Results and Perspectives

- **Results**: The model successfully estimates parameters on simulated datasets, demonstrating the feasibility of the proposed approach.
- **Future Work**: Extending the model to spatio-temporal data and scaling up for larger datasets.

## Code Structure

- **Data Simulation**: Functions to simulate Matérn Gaussian fields and construct training datasets as graphs.
- **Model Definition**: Classes for graph convolution, readout, and mapping.
- **Training and Evaluation**: Scripts to train the model and visualize results.

## Acknowledgments

This work was supervised by researchers from the MIA Paris-Saclay and the Geostatistics Laboratory at Mines Paris. Special thanks to Lucia Clarotto, Mike Pereira, Thomas Romary, and Emre Anakok for their guidance and support.

For detailed mathematical formulations, architecture diagrams, and additional context, refer to the full [French report](./RapportV3.ipynb).
