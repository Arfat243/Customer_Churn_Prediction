# Customer_Churn_Prediction
"Predicting customer churn using machine learning models like Logistic Regression and Random Forest"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (update filename if needed)
df = pd.read_csv("C:/Users/afats/Downloads/Copy of Telco_customer_churn.csv")

# Display the first few rows
print(df.head())

# Check basic information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Describe numerical columns
print(df.describe())


# Fill missing values for numerical columns with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing values for categorical columns with mode
for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify if all missing values are handled
print(df.isnull().sum())


cat_cols = ['Country', 'State', 'Gender', 'Partner', 'Dependents', 'Phone Service', 
            'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
            'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 
            'Contract', 'Paperless Billing', 'Payment Method', 'Churn Label', 'Senior Citizen']

# Apply Label Encoding directly to the dataset
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Convert 'Total Charges' to numeric directly in the dataset
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Drop NaN values if needed (optional)
df.dropna(inplace=True)

# Verify changes
print(df.info())  # Check data types
print(df.head())  # Preview dataset


# Plot Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn Value', data=df, palette="coolwarm")
plt.title("Churn Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = df.drop(columns=['Churn Value'])  # Drop target variable
y = df['Churn Value']  # Target variable

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


cat_cols = X_train.select_dtypes(include=['object']).columns
print("Categorical Columns:", cat_cols)


X_train['Total Charges'] = pd.to_numeric(X_train['Total Charges'], errors='coerce')
X_test['Total Charges'] = pd.to_numeric(X_test['Total Charges'], errors='coerce')


X_train['Total Charges'].fillna(0, inplace=True)  # Replace NaN with 0 (or use mean/median)
X_test['Total Charges'].fillna(0, inplace=True)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X_train['Senior Citizen'] = encoder.fit_transform(X_train['Senior Citizen'])
X_test['Senior Citizen'] = encoder.transform(X_test['Senior Citizen'])


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


import joblib

# Save model
joblib.dump(rf_model, "customer_churn_model.pkl")

# Load model (for later use)
loaded_model = joblib.load("customer_churn_model.pkl")

# Verify model works
print("Loaded model accuracy:", accuracy_score(y_test, loaded_model.predict(X_test)))


# Feature importance plot
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="coolwarm")
plt.title("Feature Importance in Random Forest")
plt.show()


# Compute correlation with target variable
correlations = df.corr()['Churn Value'].sort_values(ascending=False)
print(correlations)


rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Re-evaluate
print("Random Forest Accuracy after tuning:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))



