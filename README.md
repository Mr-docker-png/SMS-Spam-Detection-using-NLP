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
Dataset

The project uses the SMS Spam Collection dataset.

Dataset source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains SMS messages labeled as:

ham
spam

The original dataset contains 5,574 SMS messages.

Dataset Structure

The raw dataset contains two fields:

Column	Description
label	Target class: ham or spam
messages	SMS text

The original dataset is tab-separated.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Natural Language Processing (NLP)
TF-IDF
Jupyter Notebook
Machine Learning Models

Three classical Machine Learning algorithms were compared:

1. Multinomial Naive Bayes

Used as a baseline model for text classification.

2. Logistic Regression

A strong linear classification algorithm commonly used with TF-IDF text features.

3. Linear Support Vector Machine

A linear SVM was used because linear classifiers work effectively with high-dimensional sparse TF-IDF representations.

Text Representation

Machine Learning models cannot directly work with raw SMS text.

Therefore, the messages are converted into numerical features using:

TF-IDF

TF-IDF assigns a numerical importance score to words based on their frequency and how common they are across the document collection.

The workflow is:

SMS Text
   ↓
TF-IDF Vectorizer
   ↓
Numerical Feature Matrix
   ↓
Machine Learning Model

The TF-IDF vectorizer is fitted only on the training data and then used to transform the test data.

This prevents test-set information from leaking into the training process.

Train/Test Split

The dataset was divided into:

80% training data
20% testing data

Stratification was used to preserve the class distribution:

train_test_split(
    X,
    y,
    random_state=42,
    stratify=y,
    test_size=0.2
)
Model Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

For spam detection, spam recall and F1-score are especially important because a model that simply achieves high accuracy could still miss many spam messages.

Baseline Results
Multinomial Naive Bayes

Accuracy: 95.26%

Spam performance:

Precision: 100%
Recall: 63%
F1-score: 77%

Confusion Matrix:

[[903   0]
 [ 49  82]]
Logistic Regression

Accuracy: 96.42%

Spam performance:

Precision: 96%
Recall: 75%
F1-score: 84%

Confusion Matrix:

[[899   4]
 [ 33  98]]
Linear SVM

Accuracy: 98.16%

Spam performance:

Precision: 97%
Recall: 89%
F1-score: 92%

Confusion Matrix:

[[899   4]
 [ 15 116]]
Cross-Validation

5-fold cross-validation was performed on the training data.

For the Linear SVM:

Fold 1: 97.70%
Fold 2: 97.82%
Fold 3: 97.94%
Fold 4: 98.55%
Fold 5: 97.94%

Mean cross-validation accuracy:

97.99%

The relatively close fold scores indicate consistent performance across the validation folds.

Hyperparameter Tuning

GridSearchCV was used to tune the Linear SVM.

The parameter tested was:

C

Values tested:

[0.001, 0.01, 0.1, 1, 10, 100, 1000]
Best Parameter
C = 100
Best Cross-Validation Accuracy
98.19%
Final Model Performance

The tuned Linear SVM achieved:

Test Accuracy: 98.16%
Final Classification Report
              precision    recall  f1-score   support

ham               0.99      0.99      0.99       903
spam              0.95      0.90      0.93       131

accuracy                              0.98      1034
macro avg          0.97      0.95      0.96      1034
weighted avg       0.98      0.98      0.98      1034
Final Confusion Matrix
[[897   6]
 [ 13 118]]

This means:

897 ham messages were correctly classified.
6 ham messages were classified as spam.
118 spam messages were correctly classified.
13 spam messages were classified as ham.
Saving the Model

The trained TF-IDF vectorizer and final SVM model were saved using Joblib.

tfidf_vectorizer.pkl
sms_spam_model.pkl

This allows the trained model to be reused without retraining it every time.

Making Predictions on New SMS

A new SMS can be processed using the saved TF-IDF vectorizer and trained model.

Example: "Congratulations! You have won a free prize. Call now!"

The model predicts: spam

Another example: "Hey, are we still meeting at 6 pm?"

The model can classify it as: ham

Installation

Clone the repository:

git clone https://github.com/YOUR-USERNAME/SMS-Spam-Detection.git

Navigate into the project:

cd SMS-Spam-Detection

Install dependencies:

pip install -r requirements.txt

Run the notebook:

jupyter notebook
