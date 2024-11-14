# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv("fraudTest.csv")

# Drop unnecessary columns to reduce memory usage
data = data.drop(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
                  'street', 'city', 'state', 'zip', 'dob', 'trans_num'], axis=1)

# Initialize Label Encoder
label_encoder = LabelEncoder()

# Encode categorical columns
for column in ['merchant', 'category', 'gender', 'job']:
    data[column] = label_encoder.fit_transform(data[column].astype(str))

# Separate features and target
X = data.drop('is_fraud', axis=1)  # 'is_fraud' is the target
y = data['is_fraud']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the RandomForest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
