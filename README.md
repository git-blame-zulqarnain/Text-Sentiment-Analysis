# Sentiment Analysis on Mini IMDB Reviews

This project performs sentiment analysis on a small custom dataset of 20 movie reviews. Each review is labeled as either **positive** or **negative**. The goal is to build and evaluate a machine learning model that can classify sentiment from text data.

---

## 🔍 Dataset Description

- **File**: `mini_imdb_reviews.csv`
- **Total Reviews**: 20
- **Labels**: `positive` / `negative`
- **Source**: Custom-made for prototyping

---

## 🧠 Project Pipeline

1. **Text Preprocessing**:
   - Lowercasing, punctuation removal
   - Tokenization using regex tokenizer
   - Stopword removal
   - Lemmatization

2. **Feature Extraction**:
   - TF-IDF Vectorization

3. **Modeling**:
   - Logistic Regression

4. **Evaluation**:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization

---

## 🖥 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python sentiment_analysis.py
The output confusion matrix will be saved in visuals/confusion_matrix.png.