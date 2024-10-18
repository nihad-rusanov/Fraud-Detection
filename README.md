# Credit Card Fraud Detection Using Logistic Regression with SMOTE

This project demonstrates the detection of fraudulent credit card transactions using logistic regression. To handle the imbalance in the dataset, the project uses SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class (fraudulent transactions). The script preprocesses the data, trains a logistic regression model, and evaluates its performance using various metrics such as classification report, ROC-AUC score, and confusion matrix.

## Prerequisites

Before running this script, ensure you have the following Python libraries installed:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

You can install the required libraries using the following commands:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) available on Kaggle. You **must download the dataset** from Kaggle and place the `creditcard.csv` file in the same directory as this script in order to run it correctly.

## Overview of the Script

### 1. **Loading the Dataset**

The script loads the `creditcard.csv` dataset into a pandas DataFrame. If the file is not available, you will need to download it from Kaggle and place it in the script directory.

```python
df = pd.read_csv('creditcard.csv')
```

### 2. **Exploratory Data Analysis (EDA)**

The script performs basic exploratory data analysis (EDA):

- Displays the first few rows of the dataset.
- Checks for any missing values.
- Prints the summary statistics for the dataset.
- Visualizes the class distribution of fraudulent (Class 1) and non-fraudulent (Class 0) transactions using a count plot.

```python
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()
```

### 3. **Data Preprocessing**

- Standardizes the `Amount` and `Time` columns using `StandardScaler`.
- Drops the original `Amount` and `Time` columns and replaces them with the scaled versions.
- Rearranges the columns to ensure the scaled `Amount` and `Time` columns appear first.

```python
scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
```

### 4. **Correlation Matrix**

A correlation matrix is calculated and visualized using a heatmap to understand the relationships between the features.

```python
corr_matrix = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, cmap='coolwarm_r')
plt.title('Correlation Matrix')
plt.show()
```

### 5. **Train-Test Split**

The dataset is split into training and testing sets while maintaining the class balance using stratified splitting.

```python
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
```

### 6. **Handling Imbalanced Data with SMOTE**

To address the class imbalance in the dataset (fraudulent transactions are much fewer than non-fraudulent ones), SMOTE is applied to oversample the minority class in the training set.

```python
oversample = SMOTE(random_state=42)
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)
```

### 7. **Logistic Regression Model**

A logistic regression model is trained using the SMOTE-resampled training data. The model is then used to make predictions on the test set.

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)

y_pred_lr = lr.predict(X_test)
```

### 8. **Model Evaluation**

The script evaluates the model's performance using:

- **Classification Report**: Provides precision, recall, and F1-score for both classes.
- **ROC-AUC Score**: Measures the area under the ROC curve to evaluate the model's discriminatory ability.
- **Confusion Matrix**: Visualizes the number of true positives, false positives, true negatives, and false negatives.

```python
print('Logistic Regression Classification Report')
print(classification_report(y_test, y_pred_lr))

roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
print(f'ROC-AUC Score: {roc_auc_lr}')

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()
```

## Running the Script

1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place the `creditcard.csv` file in the same directory as this script.
2. Ensure all required Python libraries are installed by running:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. Run the script:

   ```bash
   python main.py
   ```

4. The script will print out various statistics, generate plots, and display evaluation metrics for the logistic regression model.

## Conclusion

This project demonstrates how to handle imbalanced datasets using SMOTE and how to apply logistic regression to detect fraudulent transactions. The model performance is evaluated with a variety of metrics to understand its effectiveness in classifying fraud and non-fraud transactions.
