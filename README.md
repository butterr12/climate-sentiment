# Climate Sentiment
The objective of this project is to apply sentiment analysis to an expert-labeled dataset consisting of climate-related excerpts from corporate disclosures. This analysis aims to support efforts in addressing the harmful impacts of climate change.

This repository hosts the project of Group 3, developed as part of the requirements for CS 180 during the second semester of the 2024–2025 academic year. The team is composed of Krisha Anne Chan, Judelle Gaza, and Nina Valdez
# Dataset
The dataset features climate-related paragraphs sourced from financial disclosures, primarily from large publicly listed companies. These paragraphs have been annotated by students and researchers from the University of Zurich and Friedrich-Alexander University Erlangen-Nuremberg, specializing in finance and sustainable finance. Each paragraph is labeled as either Risk, Neutral, or Opportunity. The dataset was curated by Julia Anna Bingler, Mathias Kraus, Markus Leippold, and Nicolas Webersinke

# Deployment
The project is deployed on Vercel. The dependencies needed for the web app to work is listed in requirements.txt.
# Code Structure
```bash
climate-sentiment/
├─ backend/
├─ frontend/climate-sentiment/
├─ notebooks/
│  ├─ deep/
│  │  ├─ sol1_deep.ipynb
│  │  ├─ sol2_demo_deep.ipynb
│  │  ├─ sol2_dev_deep.ipynb
│  │  ├─ sol2_predictions_deep.ipynb
│  ├─ trad/
│  │  ├─ dev_trad.ipynb
│  │  ├─ trad_demo.ipynb
│  │  ├─ trad_predictions.ipynb
├─ predictions/
│  ├─ predictions_deep.csv
│  ├─ predictions_trad.csv
├─ .gitattributes
├─ package-lock.json
├─ README.md
```
# Methodology
## Approach 1: Traditional Machine Learning (SVM)
Goal: Use TF-IDF + Support Vector Machine to classify climate sentiment.

## 📋 Preprocessing

- **Text Cleaning**: Remove punctuation, numbers, and special characters; normalize certain characters (e.g., curly quotes).
- **Lowercasing**: Convert all text to lowercase.
- **Lemmatization**: Reduce words to their base form using WordNet Lemmatizer.
- **Stopword Removal**: Eliminate common words (e.g., "the", "and").

---

## 🧠 Text Vectorization

- **TF-IDF**: Convert cleaned text into numerical features based on term frequency and inverse document frequency.
- **N-grams**: Use both unigrams and bigrams to capture more context.

---

## ⚖️ Handling Class Imbalance

- **SMOTE**: Apply Synthetic Minority Over-sampling Technique to generate synthetic examples for minority classes.

---

## 🛠 Model Training (SVM)

- **Algorithm**: Support Vector Machine with Linear, RBF, and Polynomial kernels.
- **Hyperparameter Tuning**: Use grid search to optimize parameters (e.g., `C`, kernel type).
- **Cross-Validation**: Ensure model generalization and reduce overfitting.

---

## 📊 Evaluation

- **Metrics**: Evaluate using accuracy, precision, recall, and F1-score on the dev set.
- **Retraining**: Retrain on combined training + dev set using best-found parameters.
- **Final Evaluation**: Assess performance on the unseen test set.

---

## 🔮 Prediction

- **User Input**: The final model accepts user input and outputs the predicted label with a confidence score.

---





## Approach 2: Deep Learning (Transformer - RoBERTa)
Goal: Fine-tune a pretrained RoBERTa model to classify climate sentiment.


## 📋 Preprocessing

- **Dataset**: Loads training data from JSON and validation data from CSV.
- **Tokenizer**: Uses `BertTokenizer` from HuggingFace to tokenize text with:
  - Truncation and padding to a maximum length of 128.
  - `bert-base-uncased` model for encoding text.
- **Formatting**: Datasets are formatted into PyTorch tensors with `input_ids`, `attention_mask`, and `label`.

---

## 🧠 Model Setup

- **Model**: `BertForSequenceClassification` with `bert-base-uncased` architecture.
- **Label Handling**: Dynamically sets `num_labels` based on the unique labels in the dataset.

---

## ⚙️ Training Configuration

- **Trainer API**: Uses HuggingFace's `Trainer` class.
- **TrainingArguments**:
  - Learning rate: `3e-5`
  - Batch size: `8` (train & eval)
  - Epochs: `3`
  - Evaluation and saving after every epoch
  - Outputs saved to `./results`

---

## 📊 Evaluation

- **Metrics**: Accuracy and weighted F1-score computed using:
  - `sklearn.metrics.accuracy_score`
  - `sklearn.metrics.f1_score`

---

## 💾 Saving & Prediction

- **Model Saving**: Automatically handled by HuggingFace's `Trainer`.
- **Inference**: The final model can be used to predict new user inputs using the tokenizer and model pipeline.

---
