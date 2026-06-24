# XAUUSD Robust Macro Price Forecaster

## Overview
This repository contains a production-grade, volatility-resilient LSTM time-series regression model designed to project future asset trends for Gold (`XAUUSD`). It processes a 3D tensor sequence of historical price action to generate continuous target projections 5 days into the future, serving as an execution filter and dynamic target generator within an algorithmic trading pipeline.

![visual aid](./img/lstm_visual_demonstration.png)

## Core Engineering Enhancements
* **Data Boundary Isolation:** Implements a strict sequential index firewall. Feature scaling parameters are derived exclusively from the training partition to eliminate in-sample lookahead leakage and data contamination.
* **Robust Loss Architecture:** Utilizes `nn.HuberLoss` ($\delta=1.0$) as a robust estimator. The loss function transitions from quadratic (MSE) to linear (MAE) errors during extreme price shocks, protecting network gradients from fat-tailed macro updates.
* **Adaptive Optimization:** Powered by the `Adam` optimizer to adaptively tune learning parameters across non-convex loss surfaces, replacing stagnant gradient paths.
* **Deterministic Environment:** Anchored with fixed pseudo-random state seeds across data generation layers and neural network initializations to ensure mathematical reproducibility.

## System Architecture
* **Inputs:** 3D Tensor `[Batch Size, Sequence Length (50), Features (1)]`.
* **Network Structure:** Two-layer LSTM backbone (Hidden Size: 64) mapping into a linear regression head with no activation function to accommodate un-truncated price spaces.
* **Packaging and Tooling:** Modernized under PEP 517 rules using `pyproject.toml` with `setuptools.build_meta` as the single source of truth. Formatted and linted via `black` and `ruff`.

## Execution & Verification Blueprint

### 1. Local Environment Deployment
Ensure you have Python 3.10+ installed locally.
```bash
# Clone and enter the module workspace
cd xau-huber-lstm-forecaster

# Initialize and activate an isolated virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package in editable mode along with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Run Quality Gates & Testing Suites
Execute the automated validation suite to check tensor dimensions, data engine compliance, and seed determinism:
```bash
# Run automated pytest verification
pytest test_lstm.py -v

# Run linting verification rules via Ruff
ruff check .
```

### 3. Training & Validation Execution
Execute the pipeline to train the model and trigger the out-of-sample firewall verification:
```bash
python train_lstm.py
```

### 4. Immutable Docker Deployment
To run the model within an environment isolated from native host operating system drift:
```bash
# Build the multi-stage, layer-cached production image
docker build -t xau-huber-lstm-forecaster:latest .

# Execute the containerized training suite
docker run --rm xau-huber-lstm-forecaster:latest
```

## Role in the Omni-Agent Ecosystem
This module functions as the target generator. Once a trade thesis is validated by the Volatility and Structure scanners, this LSTM provides the dynamic Take Profit (TP) and directional bias based on the macro trend memory.