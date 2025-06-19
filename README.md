# A Neural Network from Scratch

A pure Python implementation of a neural network, built from the ground up as a personal learning tool. This project avoids popular
machine learning libraries like TensorFlow or PyTorch in favor of a manual, educational approach using only NumPy and Python’s standard
library.

> **Note:** This is a personal project developed in my free time. Updates and improvements may be sporadic, but as my knowledge and
> interest in the subject grow, the pace and scope of development may change.

## Features

- Core neural network components implemented manually:
    - Forward and backward propagation
    - Activation functions (Sigmoid, ReLU, etc.)
    - Loss calculation
    - Gradient descent optimization
- Vectorized tensor operations using NumPy
- Modular, readable code designed for experimentation and learning
- Support for loading and experimenting with standard datasets (e.g., MNIST)

## Why is TensorFlow in `requirements.txt`?

TensorFlow is **only used for loading datasets** (such as MNIST) conveniently. It plays **no role** in the core neural network
implementation or algorithms. All learning, propagation, and optimization code is handcrafted using NumPy.

## Getting Started

### Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OgBreakfastBoi/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

Refer to the Python scripts in
[`src/experiments/`](https://github.com/OgBreakfastBoi/neural-network-from-scratch/tree/main/src/experiments) for example usage (e.g.,
`train_mnist.py`). You can adjust the network architecture, dataset, and training parameters as needed.

## Project Structure

- `src/nn/` — Core neural network logic and modules
- `src/data_utils/` — Dataset loading and preprocessing tools
- `src/experiments/` — Scripts for training and experimentation
- `src/visualizers/` — Tools for visualizing datasets and model behavior
- `data/` — Raw, processed, and sample datasets
- `models/` — Trained model weights, checkpoints, and serialized models
- `requirements.txt` — Project dependencies

## Contributing

This project is primarily a personal learning exercise, so I will not be accepting any pull requests. However, feedback and suggestions
are always welcome.

## License

[MIT License](LICENSE)

---

**Author:** [OgBreakfastBoi](https://github.com/OgBreakfastBoi)