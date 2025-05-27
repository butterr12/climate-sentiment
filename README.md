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

### Process:
#### Preprocessing the Text
- Lowercase all the text.
- Remove punctuation and special characters.
- Remove stopwords (like "the", "is", "and").

#### Feature Extraction
- Use TF-IDF (Term Frequency-Inverse Document Frequency) to turn text into numeric vectors.
- Set max_features (e.g., 5000 most important words).
- Use unigrams and bigrams (single words and pairs of words).
- Example: "climate change" becomes a bigram feature.

#### Train/Test Split
- Split labeled data into a training set and a development (dev) set (e.g., 80% train, 20% dev).

#### Model Training
- Train a Support Vector Machine (SVM) classifier with a linear kernel.
- Use GridSearchCV to find the best C value (regularization strength).
- Example hyperparameters: C: [0.1, 1, 10]

#### Model Evaluation
Predict on the dev set.
Calculate:
- Accuracy
- F1-Score (macro-average, so all classes are treated equally)
Choose the best model based on the F1 score.





## Approach 2: Deep Learning (Transformer - RoBERTa)
Goal: Fine-tune a pretrained RoBERTa model to classify climate sentiment.

#### Preprocessing the Text
Minimal cleaning: No need to remove stopwords or punctuation.
Keep text as natural as possible (RoBERTa handles it well).

#### Load Tokenizer and Model
Use AutoTokenizer and AutoModelForSequenceClassification from Huggingface.
Example model: "roberta-base" (or "roberta-large" for more power).
Tokenizer will:
Split sentences into word pieces (subwords).
Add [CLS] and [SEP] tokens automatically.


#### Prepare Datasets
- Tokenize the paragraphs (pad and truncate them to a max length like 256 tokens).
- Create PyTorch Dataset and DataLoader objects for training set and dev set


#### Model Fine-Tuning
Fine-tune RoBERTa on the training data.
Set training parameters (subject to change, but for now):
- Optimizer: AdamW
- Scheduler: linear scheduler with warmup
- Batch size: 16
- Epochs: 3–5
- Learning rate: 2e-5


#### Model Evaluation
After each epoch, evaluate on the dev set.
Calculate:
- Accuracy
- F1-Score (macro-average)
- Early stopping: If dev set F1 does not improve for 2 epochs, stop training.
