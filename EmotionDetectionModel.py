# EmotionDetectionModel.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


class EmotionDetectionModel:
    def __init__(self, dataset_path):
        self.data = None
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.load_data(dataset_path)
        self.preprocess_data()

    def preprocess_data(self):
        if 'Text' in self.data.columns:
            self.data['processed_text'] = self.data['Text'].apply(self.preprocess_text)
        else:
            raise ValueError("Dataset does not contain 'Text' column")

    def load_data(self, dataset_path):
        if self.data is None:
            self.data = pd.read_csv(dataset_path)

    def preprocess_text(self, text):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)

    def train_model(self):
        X = self.data['processed_text']
        y = self.data['Emotion']
        X_vectorized = self.vectorizer.fit_transform(X)
        X_train, _, y_train, _ = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)

    def predict_emotion(self, text):
        text_processed = self.preprocess_text(text)
        text_vectorized = self.vectorizer.transform([text_processed])
        return self.classifier.predict(text_vectorized)[0]
