
# Fraud Call Detection 
This project is a Fraud Call Detection app that processes audio input, converts it to text, and uses machine learning (Naive Bayes) to predict whether the call content is fraudulent or normal. The app is built using Streamlit for the UI and integrates NLTK, SpeechRecognition, and scikit-learn for preprocessing and model prediction.

## Features
- Converts spoken audio input to text using the Sppec SpeechRecognition module.
- Preprocesses the text data by removing unwanted characters, lowercasing, and lemmatizing.
- Uses CountVectorizer for text tokenization.
- Trains a Naive Bayes Classifier to predict whether a call is fraudulent or not.
- Provides accuracy score and confusion matrix for the model.

## Prerequisites
Before running the project, ensure you have the following installed on your system:

- Python 3.7+
- Pip (Python package manager)
- Required Python packages(see below)

## Required Python Packages
You can install the required packages using the following command:
```bash
pip install streamlit nltk pandas scikit-learn SpeechRecognition
```



## Installation



1. Clone my project
```bash
git clone https://github.com/yourusername/Fraud_call_detection.git
cd Fraud_call_detection
```
2. Create Virtual Environment:
It’s recommended to create a virtual environment to manage the dependencies for this project. Follow these steps to set up and activate a virtual environment:


For Windows:
```bash

cd C:\path\to\your\project
python -m venv venv
.\venv\Scripts\activate
```
For macOS/Linux:
```bash
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate
```

3. Install Requirements
```bash
pip install -r requirements.txt
```

4. If requirements.txt is not available, install the packages manually:
```bash
pip install streamlit nltk pandas scikit-learn SpeechRecognition
```
5. Dataset
```bash
Ensure the file Fraud_calls.txt is in your project directory. 
```
6.Run the app 
```bash
streamlit run main.py
```



## How It Works
1. Audio Input: The app listens to a short audio clip using your microphone.
2. Speech to Text: The audio is converted to text using Google’s Speech Recognition API.
3. Text Preprocessing: The text is cleaned and lemmatized using NLTK.
4. Prediction: The processed text is passed to the trained Naive Bayes classifier, which predicts whether the call is fraudulent or not.
5. Results: The app displays the predicted label (fraud or normal) and the corresponding accuracy of the model.

## Tech Stack

**Language** : Python


**Libraries** :  Speech Recognition, Streamlit, Pandas, Numpy, NLTK


# Fraud-Call-Detection
