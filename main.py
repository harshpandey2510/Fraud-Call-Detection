import streamlit as st
import nltk
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the speech recognizer
recognizer = sr.Recognizer()


# Function to convert audio to text
def audio_to_text():
    conv = []
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        st.write("Listening stopped")
    try:
        text = recognizer.recognize_google(audio)
        conv.append(text)
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand.")
    return conv


# Method to preprocess the data
def preprocessing(dataset, num_of_rows=1):
    stemmer = WordNetLemmatizer()
    corpus = []
    for i in range(0, num_of_rows):
        document = re.sub(r'\W', ' ', dataset[i])
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(w) for w in document]
        document = ' '.join(document)
        corpus.append(document)
    return corpus


# Method to train model and predict
def predict_using_count_vectorizer(dataset, num_of_rows, callData):
    x = preprocessing(dataset['call_content'], num_of_rows)
    y = dataset.Label
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    live_test = preprocessing(callData, len(callData))

    count_vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    count_train = count_vectorizer.fit_transform(x_train)
    count_test = count_vectorizer.transform(x_test)
    live_count_test = count_vectorizer.transform(live_test)

    nbclassifier = MultinomialNB()
    nbclassifier.fit(count_train, y_train)
    y_pred = nbclassifier.predict(count_test)
    live_y_pred = nbclassifier.predict(live_count_test)
    score = metrics.accuracy_score(y_test, y_pred)

    cm = metrics.confusion_matrix(y_pred, y_test, labels=['normal', 'fraud'])
    st.write("Accuracy score when using count vectorizer: ", score)
    st.write("CONFUSION MATRIX\n", cm)

    return callData, live_y_pred


# Streamlit UI
st.title("Fraud Call Detection")
st.write("Press the button below to start recording your call content.")

if st.button("Record Audio"):
    callData = audio_to_text()
    if callData:
        # Load training dataset
        dataset = pd.read_csv(r"Fraud_calls.txt", sep='|')
        num_of_rows, _ = dataset.shape
        conversation, label = predict_using_count_vectorizer(dataset, num_of_rows, callData)

        st.write("Audio Received: {}".format(conversation))
        st.write("Predicted Label: {}".format(label[0]))  # Assuming single label prediction
