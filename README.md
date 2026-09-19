# SMS Spam Detection using NLP

A Machine Learning project that detects whether an SMS message is **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and classical Machine Learning algorithms.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text messages into numerical features and compares multiple Machine Learning models before selecting and tuning a Linear Support Vector Machine (SVM).

---

## Project Overview

SMS spam is a common problem where unwanted promotional or fraudulent messages are sent to users.

In this project, an NLP-based Machine Learning system is developed to automatically classify SMS messages into two categories:

- **Ham** — legitimate SMS
- **Spam** — unwanted/spam SMS

The project follows a complete Machine Learning workflow:

```text
Raw SMS Dataset
      ↓
Data Exploration
      ↓
Data Cleaning
      ↓
Train/Test Split
      ↓
TF-IDF Vectorization
      ↓
Model Training
      ↓
Model Evaluation
      ↓
5-Fold Cross-Validation
      ↓
GridSearchCV
      ↓
Final Tuned Model
      ↓
New SMS Prediction
