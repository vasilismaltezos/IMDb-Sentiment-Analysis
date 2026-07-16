# IMDb Sentiment Analysis

## Overview
This project implements and compares classical Machine Learning and Deep Learning approaches for binary sentiment classification on the IMDb Large Movie Review Dataset.

The project was developed as part of an Artificial Intelligence course and includes both traditional machine learning algorithms and recurrent neural networks implemented in Python.

## Features

### Part A - Machine Learning
- Naive Bayes
- Logistic Regression
- Random Forest
- Feature Extraction
- Learning Curve Evaluation

### Part B - Deep Learning
- Bidirectional RNN
- LSTM/GRU
- PyTorch implementation
- Model training and evaluation

## Dataset

- IMDb Large Movie Review Dataset
- Binary sentiment classification (Positive / Negative)

## Technologies

- Python
- Scikit-learn
- PyTorch
- NumPy
- Matplotlib

## Project Structure

```
.
├── main.py
├── preprocessing.py
├── feature_extraction.py
├── naive_bayes.py
├── logistic_regression.py
├── imdb_rf_tuned.py
├── imdb_rnn.py
├── evaluation.py
├── learning_curves.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

Classical Machine Learning:

```bash
python main.py
```

Deep Learning:

```bash
python imdb_rnn.py
```

## Results

The project evaluates different machine learning and deep learning models using:

- Precision
- Recall
- F1-score
- Learning Curves
