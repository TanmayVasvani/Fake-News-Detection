import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
from flask import Flask, request, jsonify

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
def load_dataset():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv"
    df = pd.read_csv(url)
    return df

# Preprocess Data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Preprocess Dataset
def preprocess_dataset(df):
    df['text'] = df['text'].apply(preprocess_text)
    return df

# Train Model
def train_model(df):
    X = df['text']
    y = df['label'].map({'FAKE': 0, 'REAL': 1})

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model, vectorizer

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json.get('news')
    if not news:
        return jsonify({'error': 'No news text provided'}), 400

    processed_news = preprocess_text(news)
    vectorized_news = vectorizer.transform([processed_news])
    prediction = model.predict(vectorized_news)
    result = "FAKE" if prediction < 0.5 else "REAL"
    return jsonify({'prediction': result})

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset()
    dataset = preprocess_dataset(dataset)

    print("Training model...")
    model, vectorizer = train_model(dataset)

    model.save('fake_news_model.h5')
    print("Model saved as 'fake_news_model.h5'.")

    print("Starting Flask app...")
    app.run(debug=True, port=5000)