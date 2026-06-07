# Epidemic Transformer Forecasting

This repository contains the code for an influenza-like illness (ILI) forecasting project using Transformer-based time series models.

The project reproduces a Transformer forecasting model and compares it with several baselines, including Persistence Forecasting, ARIMA, and LSTM. It also explores three improvements: horizon-weighted loss, multivariate inputs, and a modern decoder-only Transformer.

## Repository Structure

```text
.
├── data/
│   ├── state/
│   └── national/
├── transformer-influenza-predicting.ipynb.ipynb
├── README.md
└── ...
```

## Dataset

The experiments use CDC ILINet and laboratory surveillance data.

The forecasting task is:

- Input history window: 10 weeks
- Prediction horizon: 4 weeks

The dataset is split chronologically into training, validation, and test sets.

## Setup

Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib scipy scikit-learn statsmodels jupyter
```

## Running the Code

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open and run:

```text
transformer-influenza-predicting.ipynb
```

Run the notebook cells from top to bottom to reproduce the main experiments.

## Models Included

The notebook includes:

- Persistence Forecasting
- ARIMA
- LSTM
- Original encoder-decoder Transformer
- Horizon-weighted loss Transformer
- Multivariate Transformer
- Modern decoder-only Transformer

## Reproducing Results

To reproduce the main results:

1. Place the CDC data files under the `data/` directory.
2. Open `transformer-influenza-predicting.ipynb`.
3. Run the data loading and preprocessing cells.
4. Run the baseline model cells.
5. Run the Transformer and improvement model cells.
6. Run the evaluation and visualization cells.

The notebook reports RMSE and Pearson correlation for each model.
