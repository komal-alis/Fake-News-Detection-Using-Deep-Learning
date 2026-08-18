📰 Fake News Detection Using Deep Learning

A deep learning-based Fake News Detection System that uses Natural Language Processing (NLP), TF-IDF feature extraction, and a trained Neural Network to classify news articles as REAL or FAKE.

📌 Project Overview

Fake news spreads quickly through digital platforms and can mislead people. This project provides a simple web-based system that analyzes news text and predicts whether it is real or fake.

The application is developed using Python and Streamlit and uses a pre-trained deep learning model for prediction.

✨ Features

📰 Detects REAL or FAKE news
🤖 Uses a trained Deep Learning model
🔤 Text preprocessing using NLP
📊 TF-IDF-based feature extraction
🎯 Displays prediction score and confidence
🌐 Simple and interactive Streamlit interface
🧪 Includes example news for testing

🛠️ Technologies Used

Python
TensorFlow / Keras
Streamlit
Natural Language Processing (NLP)
TF-IDF
Scikit-learn
NLTK
Pandas
NumPy

📂 Project Structure

Fake-News-Detection-Using-Deep-Learning/
│
├── .gitignore
├── README.md
├── app_streamlit.py
├── model.h5
├── requirements.txt
└── vectorizer.pkl

File Description

| File               | Description                      |
| ------------------ | -------------------------------- |
| `.gitignore`       | Files and folders ignored by Git |
| `README.md`        | Project documentation            |
| `app_streamlit.py` | Main Streamlit application       |
| `model.h5`         | Trained deep learning model      |
| `requirements.txt` | Required Python libraries        |
| `vectorizer.pkl`   | Saved TF-IDF vectorizer          |


🔄 How It Works

News Text
↓
Text Preprocessing
↓
Stopword Removal & Lemmatization
↓
TF-IDF Vectorization
↓
Deep Learning Model
↓
Prediction
↓
REAL / FAKE

The application first cleans the entered news text using NLP techniques. The cleaned text is then converted into numerical features using the saved TF-IDF vectorizer. Finally, the trained neural network predicts whether the news is REAL or FAKE.

🚀 Installation

Clone the repository

git clone https://github.com/komal-alis/Fake-News-Detection-Using-Deep-Learning.git

Move into the project directory

cd Fake-News-Detection-Using-Deep-Learning

Install the required libraries

pip install -r requirements.txt

▶️ Run the Application

Start the Streamlit application using:

streamlit run app_streamlit.py

The application will open in your browser.

🧠 Model Information

Algorithm: Deep Learning
Feature Extraction: TF-IDF
Model: Deep Neural Network
Training Epochs: 15
Dataset: Fake.csv + True.csv + Additional Dataset

📊 Prediction

The system provides:

REAL NEWS or FAKE NEWS prediction
Prediction score
Confidence percentage

🧪 Ablation Study

An ablation study was performed to evaluate the effect of different preprocessing techniques, feature configurations, dropout settings, and model architectures on the model's performance.

The final proposed model achieved approximately 85% accuracy.

🎯 Future Improvements

Improve model accuracy with a larger and more diverse dataset
Add more advanced NLP techniques
Explore multiple deep learning architectures
Deploy the application online
Add support for multiple languages
Improve the user interface

👩‍💻 Author
Astha Shrivastava
Komal Ojha

GitHub: @komal-alis

📄 License

This project is created for educational and project purposes.
