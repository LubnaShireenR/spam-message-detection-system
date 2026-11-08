
# 📩 Spam Message Detection System

### An AI-Powered Text Classification Application Using Machine Learning and NLP

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

---

## 📌 Project Overview

The Spam Message Detection System is a supervised machine learning project that identifies whether a text message is Spam or Ham (Legitimate).

The project uses Natural Language Processing (NLP), TF-IDF vectorization, and machine learning classification algorithms to analyse message content and predict its category.

A Streamlit web application provides an interactive interface where users can enter messages and receive predictions.

## 🎯 Project Objectives

- Develop an automated spam message classification system.
- Apply NLP techniques to clean and preprocess text data.
- Convert text messages into numerical features using TF-IDF.
- Train and compare multiple machine learning algorithms.
- Evaluate model performance using classification metrics.
- Deploy the trained model through a Streamlit web application.

## 📂 Dataset

**Dataset:** SMS Spam Collection Dataset

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains labelled SMS messages classified into two categories:

| Label | Description |
|---|---|
| Ham | Legitimate messages |
| Spam | Unwanted or fraudulent messages |

The loaded dataset contains 5,572 messages.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Natural Language Processing | NLTK |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Algorithms | Naive Bayes, Logistic Regression, SVM |
| Model Evaluation | Accuracy, Precision, Recall, F1-score |
| Deployment | Streamlit |
| Model Serialization | Joblib |
| Version Control | Git, GitHub |

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Loading and Inspection
3. Exploratory Data Analysis (EDA)
4. Spam and Ham Distribution Analysis
5. Message Length Analysis
6. Text Cleaning and Preprocessing
7. Tokenization and Stemming
8. TF-IDF Feature Extraction
9. Training and Testing Data Split
10. Machine Learning Model Development
11. Model Performance Comparison
12. Confusion Matrix Analysis
13. Model Selection
14. Model Saving
15. Streamlit Application Development

## 🤖 Machine Learning Algorithms

Three supervised classification algorithms were implemented:

### 1. Multinomial Naive Bayes
A probabilistic classification algorithm commonly used for text classification.

### 2. Logistic Regression
A classification algorithm that estimates the probability of a message belonging to a particular category.

### 3. Support Vector Machine (SVM)
A classification algorithm that identifies an optimal decision boundary between spam and legitimate messages.

## 📊 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The final model is selected based on its F1-score, with particular attention to spam-class recall.

## 🧹 NLP Preprocessing

The following text preprocessing operations are performed:

- Lowercase conversion
- URL removal
- Email address removal
- Punctuation and number removal
- Stopword removal
- Tokenization
- Porter Stemming

## 🔢 TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) converts text messages into numerical feature representations.

The vectorizer uses:
- Maximum features: 5,000
- N-gram range: (1, 2)

The vectorizer is fitted only on the training data to prevent data leakage.

## 🌐 Streamlit Web Application

The project includes an interactive dark-themed Streamlit application.

### Application Features

- Spam and Ham message prediction
- Interactive message input
- Example spam and legitimate messages
- Word count and character count
- Prediction result cards
- Machine learning model integration
- Project information sidebar

## 📁 Repository Structure

```text
spam-message-detection-system/
│
├── dataset/
│   ├── SMSSpamCollection
│   └── readme
│
├── models/
│   ├── spam_detection_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── email_spam_detection_system.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/LubnaShireenR/spam-message-detection-system.git
```

### Step 2: Navigate to the project directory

```bash
cd spam-message-detection-system
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch Streamlit

```bash
streamlit run app.py
```

### Step 5: Open the application

```text
http://localhost:8501
```

## 📈 Key Learning Outcomes

- Understanding supervised machine learning.
- Practical implementation of NLP.
- Text preprocessing and feature extraction.
- Training and comparing classification models.
- Evaluating classification performance.
- Saving and loading trained ML models.
- Building an interactive application using Streamlit.

## 🔮 Future Improvements

- Integrate a larger email spam dataset.
- Improve model generalization.
- Add probability calibration and confidence estimates.
- Introduce email-specific spam classification.
- Deploy the application publicly.
- Add model performance monitoring.

## 👩‍💻 Author

**Lubna Shireen R**

B.Tech – Electronics and Communication Engineering  
Specialization: Data Science

GitHub: https://github.com/LubnaShireenR

LinkedIn: https://www.linkedin.com/in/lubnashireenr

---

⭐ If you find this project interesting, consider giving the repository a star!
