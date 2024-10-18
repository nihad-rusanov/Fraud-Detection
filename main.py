import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression

# Imbalanced Data Handling
from imblearn.over_sampling import SMOTE



# Load the dataset
df = pd.read_csv('creditcard.csv')

# # Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())


#Class distribution
print(df['Class'].value_counts())

#Visualize the class distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# # Rearrange columns
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)



# Correlation matrix
corr_matrix = df.corr()

#Plot the heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap='coolwarm_r')
plt.title('Correlation Matrix')
plt.show()

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# SMOTE for oversampling minority class
oversample = SMOTE(random_state=42)

X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)

# Check the new class distribution
#print(pd.Series(y_train_smote).value_counts())



lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_lr = lr.predict(X_test)

print('Logistic Regression Classification Report')
print(classification_report(y_test, y_pred_lr))

roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f'ROC-AUC Score: {roc_auc_lr}')

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()



