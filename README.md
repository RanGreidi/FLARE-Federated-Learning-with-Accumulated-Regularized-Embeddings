# FLARE-Federated-Learning-with-Accumulated-Regularized-Embeddings

This repository contains code, data, and documentation for our research project on federated learning. The project explores [brief description of your research focus].

## Table of Contents

- [Abstract](#Abstract)
- [Directory Structure](#directory-structure)
- [Setup and Dependencies](#setup-and-dependencies)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Abstract

Federated Learning (FL) is an emerging paradigm that allows for decentralized machine learning (ML), where multiple models are collaboratively trained in a privacy-preserving manner. It has attracted much interest due to the significant advantages it brings to training deep neural network (DNN) models, particularly in terms of prioritizing privacy and enhancing the efficiency of communication resources when local data is stored at the edge devices. However, since communications and computation resources are limited, training DNN models in FL systems face challenges such as elevated computational and communication costs in complex tasks.

Sparse training schemes gain increasing attention in order to scale down the dimensionality of each client (i.e., node) transmission. Specifically, sparsification with error correction methods is a promising technique, where only important updates are sent to the parameter server (PS) and the rest are accumulated locally. While error correction methods have shown to achieve a significant sparsification level of the client-to-PS message without harming convergence, pushing sparsity further remains unresolved due to the staleness effect. In this paper, we propose a novel algorithm, dubbed Federated Learning with Accumulated Regularized Embeddings (FLARE), to overcome this challenge. FLARE presents a novel sparse training approach via accumulated pulling of the updated models with regularization on the embeddings in the FL process, providing a powerful solution to the staleness effect, and pushing sparsity to an exceptional level. The performance of FLARE is validated through extensive experiments on diverse and complex models, achieving a remarkable sparsity level (10 times and more beyond the current state-of-the-art) along with significantly improved accuracy. Additionally, an open-source software package has been developed for the benefit of researchers and developers in related fields.

## Directory Structure

Here's an overview of the project's directory structure:

- `data/`: Raw and processed data files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experiments.
- `src/`: Python source code for data preprocessing, model training, and evaluation.
- `src/federated_learning/`: Code specific to federated learning implementation.
- `experiments/`: Configuration files, results, and logs for different experiments.
- `models/`: Saved model files.
- `requirements.txt`: List of required Python packages.
- `README.md`: Documentation providing an overview of the project.

## Setup and Dependencies

To set up the project environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

## Results
![Project Image](Results/)
