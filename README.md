# Reddit Sentiment Analysis using Machine Learning and Deep Learning

## Overview

This project presents an end-to-end Natural Language Processing (NLP) pipeline for sentiment classification of Reddit comments. The workflow covers data preprocessing, exploratory data analysis, feature engineering, traditional machine learning models, and deep learning architectures for multi-class sentiment prediction.

The primary objective is to compare classical machine learning approaches with deep learning models and identify the most effective solution for sentiment analysis.

---

## Project Structure

```text
├── 01_eda_preprocessing.ipynb
├── 02_baseline_model.ipynb
├── 03_deeplearning_models.ipynb
├── README.md
└── requirements.txt
```

---

## Dataset

The dataset consists of Reddit comments labeled into three sentiment categories:

- Negative
- Neutral
- Positive

The project includes extensive preprocessing to transform noisy user-generated text into a format suitable for machine learning and deep learning models.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing duplicate records
- Text cleaning and normalization
- Lowercasing
- Tokenization

All preprocessing steps are documented in:

```text
01_eda_preprocessing.ipynb
```

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand dataset characteristics and identify potential challenges before model development.

Analysis includes:

- Missing value analysis
- Duplicate analysis
- Sentiment distribution
- Comment length distribution
- Most frequent words
- Text statistics and visualizations

---

## Feature Engineering

### Traditional Machine Learning

Text data was transformed using:

- TF-IDF Vectorization
- Unigrams and Bigrams
- Hyperparameter tuning using GridSearchCV

### Deep Learning

Text sequences were prepared using:

- Tokenization
- Vocabulary creation
- Sequence padding
- Embedding layers

---

## Machine Learning Models

The following machine learning algorithms were trained and evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Random Forest
- Linear Support Vector Machine (SVM)

### Best Machine Learning Model

Linear SVM achieved the highest performance among all traditional machine learning models.

#### Linear SVM Classification Report

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Negative | 0.83 | 0.69 | 0.76 |
| Neutral | 0.84 | 0.95 | 0.89 |
| Positive | 0.89 | 0.87 | 0.88 |

**Overall Accuracy:** 85.45%  
**Weighted F1 Score:** 85.19%

Notebook:

```text
02_baseline_model.ipynb
```

---

## Deep Learning Models

### Model 1: Embedding + Dense Neural Network

Architecture:

```text
Embedding Layer
        ↓
GlobalAveragePooling1D
        ↓
Dense Layer
        ↓
Dropout
        ↓
Softmax Output Layer
```

### Model 2: Embedding + Dense Network with Class Weights

The same architecture was trained using class weights to address class imbalance.

### Model 3: Bidirectional LSTM

Architecture:

```text
Embedding Layer
        ↓
SpatialDropout1D
        ↓
Bidirectional LSTM
        ↓
Dense Layer
        ↓
Dropout
        ↓
Softmax Output Layer
```

Features:

- Sequence modeling
- Context capture from both directions
- Early stopping
- Learning rate scheduling
- Class weighting
- Model checkpointing

Notebook:

```text
03_deeplearning_models.ipynb
```

---

## Model Comparison

| Model | Accuracy |
|---------|---------|
| Naive Bayes | 62.06% |
| Logistic Regression | 83.08% |
| Random Forest | 79.95% |
| Embedding + Dense Network | 83.62% |
| Embedding + Dense Network (Class Weights) | 84.06% |
| Bidirectional LSTM | 84.37% |
| Linear SVM | 85.45% |

### Key Findings

- Linear SVM achieved the best overall performance with an accuracy of 85.45%.
- Deep learning models achieved competitive performance, with Bidirectional LSTM reaching 84.37% accuracy.
- Applying class weights improved neural network performance from 83.62% to 84.06%.
- TF-IDF combined with Linear SVM proved highly effective for sentiment classification on this dataset.

---

## Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

### NLP Techniques

- Text Cleaning
- TF-IDF Vectorization
- Tokenization
- Word Embeddings
- Sequence Padding

---

## Workflow

```text
Raw Reddit Comments
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Machine Learning Models
        ↓
Deep Learning Models
        ↓
Evaluation and Comparison
```

---

## Trained Models

Trained model files are not included in this repository due to GitHub file size limitations.

The models can be reproduced by running the notebooks in the following order:

```text
1. 01_eda_preprocessing.ipynb
2. 02_baseline_model.ipynb
3. 03_deeplearning_models.ipynb
```

---

## Future Work

- Fine-tuning transformer-based models such as DistilBERT and BERT
- Hyperparameter optimization
- Cross-validation experiments
- Model deployment using FastAPI
- Real-time sentiment analysis applications

