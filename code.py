import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('loan_prediction.csv')

# Drop irrelevant columns
df = df.drop('Loan_ID', axis=1)

# Handling missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Remove outliers based on IQR
Q1 = df['ApplicantIncome'].quantile(0.25)
Q3 = df['ApplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['ApplicantIncome'] >= (Q1 - 1.5 * IQR)) & (df['ApplicantIncome'] <= (Q3 + 1.5 * IQR))]

Q1 = df['CoapplicantIncome'].quantile(0.25)
Q3 = df['CoapplicantIncome'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['CoapplicantIncome'] >= (Q1 - 1.5 * IQR)) & (df['CoapplicantIncome'] <= (Q3 + 1.5 * IQR))]

# Convert categorical columns to numerical using one-hot encoding
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=cat_cols)

# Split the dataset into features (X) and target (y)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Convert Loan_Status to numeric values if needed (e.g., 1 for 'Y', 0 for 'N')
y = y.map({'Y': 1, 'N': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical columns using StandardScaler
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols].copy())
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols].copy())

# Train a Support Vector Classifier
model = SVC(random_state=42)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Convert X_test to a DataFrame and add predictions
X_test_df = pd.DataFrame(X_test, columns=X_test.columns)
X_test_df['Loan_Status_Predicted'] = y_pred
print(X_test_df.head())
