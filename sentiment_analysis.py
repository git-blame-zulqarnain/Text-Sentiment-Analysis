import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

os.makedirs('plots', exist_ok=True)
path=os.path.join("dataset","mini_imdb_reviews.csv")
df=pd.read_csv(path)

print("Sample data:\n",df.head())

def preprocess_text(text):

    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text)
    tokens=nltk.word_tokenize(text)
    
    lemmatizer= WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(cleaned_tokens)
df['clean_review']=df['review'].apply(preprocess_text)

tfidf=TfidfVectorizer()
x=tfidf.fit_transform(df['clean_review'])
y=df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("Visuals/confusion_matrix.png")
plt.clf()