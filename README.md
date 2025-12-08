# Logistic Regression --- From Scratch (Python)

This repository contains a clean, from-scratch implementation of
**Logistic Regression** using only fundamental Python libraries (NumPy,
Matplotlib, etc.).\
The project demonstrates how to train a binary classifier, visualize
training progress, and plot decision boundaries without using
machine-learning frameworks such as scikit-learn.

## Project Structure

    .
    ├── datasets/
    │   └── ...        # Training/testing datasets
    ├── results/
    │   └── ...        # Saved plots, metrics, and model outputs
    ├── logistic_regression.py   # Core Logistic Regression implementation
    ├── main.py                  # Entry point for training/testing
    ├── options.py               # Hyperparameter + CLI configuration
    ├── plot.py                  # Loss, accuracy, and decision boundary plots
    └── README.md                # Project documentation

## Features

-   Gradient-based training (batch gradient descent)
-   Sigmoid hypothesis function
-   Cross-entropy loss
-   L2 regularization support
-   Accuracy tracking during training
-   Loss/accuracy visualization
-   2D decision boundary plotting
-   Modular codebase

## Requirements

Install dependencies:

``` bash
pip install numpy matplotlib pandas
```

Python version:

-   **Python 3.9+** recommended

## Usage

Run training:

``` bash
python main.py
```

With custom hyperparameters:

``` bash
python main.py --lr 0.1 --iterations 5000 --lambda 0.01
```

Available arguments (see `options.py`):

  Argument         Description                      Default
  ---------------- -------------------------------- ---------
  `--lr`           Learning rate                    0.01
  `--iterations`   Number of training iterations    1000
  `--lambda`       L2 regularization coefficient    0.0
  `--plot`         Enable plotting after training   true

## Visualizations

The repository automatically generates:

-   Loss vs. Iterations plot\
-   Accuracy vs. Iterations plot\
-   Decision boundary (if dataset is 2D)

Generated figures are saved into the `results/` directory.

## Core Idea

The model predicts probabilities using the **sigmoid** function:

\[ h\_`\theta`{=tex}(x) = `\frac{1}{1 + e^{-(w^T x)}}`{=tex} \]

And minimizes the **regularized cross-entropy loss**:

\[ J(w) = -`\frac{1}{m}`{=tex}`\sum`{=tex}\_{i=1}\^m
`\left[ y_i\log(h(x_i)) + (1 - y_i)\log(1 - h(x_i)) \right] `{=tex}+
`\frac{\lambda}{2m}`{=tex}\|w\|\_2\^2 \]

Gradients are derived analytically and updated using gradient descent.

## Datasets

Place your datasets inside the `datasets/` folder.

Supported formats:

-   CSV\
-   TXT with numeric columns

If you use custom data, ensure:

-   Last column = label (0 or 1)
-   All features are numeric

## Notes

-   This implementation is strictly educational --- it is intentionally
    simple and fully transparent.
-   No high-level ML libraries are used.

## License

MIT License. Feel free to use, modify, and contribute.
