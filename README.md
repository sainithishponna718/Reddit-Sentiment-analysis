# Reddit Comment Sentiment Analysis
 
A four-part project exploring sentiment classification on Reddit comments, progressing from classical ML through deep learning to fine-tuned transformers — comparing approaches on the same dataset to see how much each step up in model sophistication actually buys you.
 
## Dataset
 
[Reddit Sentiment Analysis dataset](https://github.com/Himanshu-1703/reddit-sentiment-analysis) — ~37k Reddit comments labeled across three sentiment classes:
 
| Label | Category | Share |
|---|---|---|
| `-1` | Negative | ~22% |
| `0` | Neutral | ~35% |
| `1` | Positive | ~43% |
 
**Note on data quality:** the `clean_comment` column was already stopword-stripped by the original dataset author before this project began (verified directly against the upstream source). This is a known constraint, especially for the transformer stage — transformers benefit most from natural sentence structure, and that wasn't fully available here.
 
## Project structure
 
| Notebook | Description |
|---|---|
| `01_eda_preprocessing.ipynb` | Data cleaning (nulls, duplicates, whitespace), exploratory analysis, stopword removal (preserving negation words like "not"), lemmatization |
| `02_baseline_model.ipynb` | TF-IDF + classical ML models (Naive Bayes, Logistic Regression, Random Forest, Linear SVM), GridSearchCV tuning |
| `03_deeplearning_models.ipynb` | Keras Tokenizer + embeddings, feedforward ANN (with and without class weighting), Bidirectional LSTM |
| `04_transformer_finetuning.ipynb` | DistilBERT fine-tuning (PyTorch + Hugging Face `Trainer`), Optuna hyperparameter search |
 
## Results
 
| Model | Accuracy | Macro F1 |
|---|---|---|
| Linear SVM (TF-IDF, tuned) | 0.86 | 0.84 |
| ANN | 0.84 | 0.82 |
| ANN (class-weighted) | 0.84 | 0.83 |
| Bidirectional LSTM | 0.84 | 0.83 |
| DistilBERT (baseline) | 0.93 | 0.92 |
| **DistilBERT (Optuna-tuned)** | **0.94** | **0.93** |
 
Fine-tuned DistilBERT outperforms every classical and from-scratch deep learning baseline by roughly 8-10 points — a substantial jump, even working from the stopword-stripped input that's suboptimal for transformer models.
 
### Where tuning helped most
 
Optuna tuning (15 trials, optimizing validation macro-F1) improved the negative class specifically — the class that struggled across every single model in this project:
 
| | Baseline DistilBERT | Tuned DistilBERT |
|---|---|---|
| Negative recall | 0.85 | 0.89 |
| Negative misclassifications | 182 / 1238 | 140 / 1238 |
 
Best hyperparameters found: learning rate `4.43e-05`, batch size `32`, `4` epochs, warmup ratio `0.04`, weight decay `0.026`. Across all 15 trials, higher learning rates (3e-05 to 5e-05) and more epochs consistently produced the best results — low learning rates underperformed even with extra epochs to compensate.
 
### Remaining limitations
 
The dominant error pattern, even after tuning, is negative↔positive confusion rather than negative↔neutral — suggesting some comments carry genuinely ambiguous or sarcastic sentiment that's hard to resolve without full negation/context cues, likely compounded by the missing stopwords in the source text.
 
## Setup
 
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```
 
GPU is required for practical training time on notebook 04 (run on Kaggle/Colab with a T4 or better). PyTorch installs as CPU-only by default via pip; for GPU training, install the CUDA build per [pytorch.org](https://pytorch.org/get-started/locally/).
 
## Model weights
 
Trained model weights are not included in this repo due to size (DistilBERT checkpoints are ~250MB). The notebook can be re-run to reproduce them, or the model can be hosted separately on Hugging Face Hub if needed.
 
## Tech stack
 
- **Classical ML:** scikit-learn (TF-IDF, Linear SVM, GridSearchCV)
- **Deep learning:** TensorFlow / Keras (ANN, BiLSTM)
- **Transformers:** PyTorch, Hugging Face `transformers` + `datasets` + `Trainer`
- **Hyperparameter tuning:** Optuna
