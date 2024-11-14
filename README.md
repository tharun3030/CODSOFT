# CODSOFT

## Task 2: Credit Card Fraud Detection

This project aims to build a machine learning model to detect fraudulent credit card transactions. Using the provided dataset, the model classifies each transaction as either legitimate or fraudulent, helping financial institutions prevent fraud and reduce losses.

## Dataset

The dataset used for this project can be found on Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

This dataset contains various features from credit card transactions, including transaction amounts, times, and anonymized customer information. Each transaction is labeled as either `fraudulent` or `legitimate`.

## Features

- **Data Preprocessing**: Prepares the data for training by handling missing values, scaling features, and splitting data into training and testing sets.
- **Model Selection**: Implements a Random Forest classifier to predict fraud.
- **Evaluation**: Measures the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place it in the project directory.
2. Run the `credit_card_fraud_detection.py` file:
   ```bash
   python credit_card_fraud_detection.py
   ```

This script will preprocess the data, train the model, and output performance metrics.

## Model Evaluation

The Random Forest classifier’s performance is evaluated with the following metrics:
- **Accuracy**: Measures the percentage of correctly classified transactions.
- **Precision**: Indicates the proportion of actual fraud cases among the detected cases.
- **Recall**: Measures how many fraud cases were correctly identified.
- **F1-Score**: Provides a balanced measure of precision and recall.

## Example Output

Upon running the script, you will see output similar to the following:

```
Accuracy: 0.99
Classification Report:
               precision    recall  f1-score   support
    Legitimate       0.99      1.00      0.99      1000
       Fraudulent     0.95      0.90      0.92       50

Confusion Matrix:
[[998   2]
 [  5  45]]
```

## Task 3: Customer Churn Prediction

**Project Overview**  
This project involves building a predictive model to identify customers who are at risk of churning from a subscription-based service. Using historical customer data, the model aims to capture patterns of usage behavior and demographics to understand factors that contribute to customer churn. This information can help businesses improve retention strategies and address customer needs proactively.

**Technologies Used**  
- Python
- Machine Learning Algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Data Preprocessing and Feature Engineering
- Evaluation Metrics: Precision, Recall, F1-score, ROC-AUC

**Dataset**  
The dataset contains customer demographic information and usage patterns. It can be found at https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction.

**Key Highlights**  
- Preprocessed the dataset by handling missing values and encoding categorical data.
- Built multiple models and compared their performance to find the most accurate and reliable model.
- Evaluated model performance using accuracy metrics, helping to interpret the results for business insights.


## Task 4: Spam SMS Detection

**Project Overview**  
This project focuses on developing an AI model to classify SMS messages as either spam or legitimate. The goal is to create a system that can filter out unwanted spam messages and improve user experience in messaging applications. The model leverages natural language processing techniques to analyze message content and predict its category.

**Technologies Used**  
- Python
- Natural Language Processing (NLP): TF-IDF, Word Embeddings
- Machine Learning Algorithms: Naive Bayes, Logistic Regression, Support Vector Machines
- Evaluation Metrics: Precision, Recall, F1-score

**Dataset**  
The dataset consists of SMS messages labeled as spam or ham (legitimate) and can be found at https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset.

**Key Highlights**  
- Performed text preprocessing, including tokenization and vectorization using TF-IDF.
- Implemented multiple classification algorithms to identify the most effective model for spam detection.
- Evaluated models using metrics like precision and recall to ensure high accuracy in spam classification.
