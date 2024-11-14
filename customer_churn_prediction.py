# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Preview the data to understand its structure
print("Dataset Preview:")
print(data.head())
print("\nData Columns:", data.columns)

# Assuming 'Exited' is the target column for churn, renaming it to 'Churn' for consistency
data = data.rename(columns={'Exited': 'Churn'})

# Drop irrelevant columns (like 'RowNumber', 'CustomerId', and 'Surname')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables to dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
