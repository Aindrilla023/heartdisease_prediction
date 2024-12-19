import numpy as np
import pandas as pd
data = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')
print(data.head())
from google.colab import files
import numpy as np
import pandas as pd

# Number of rows and columns
num_rows = data.shape[0]
num_cols = data.shape[1]

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

# Range of the dataset (assuming numerical columns)
numerical_cols = data.select_dtypes(include=np.number).columns

for col in numerical_cols:
  print(f"Range of {col}: Min={data[col].min()}, Max={data[col].max()}")


# prompt: how many attribute are there and how many patients records are there

from google.colab import files
import numpy as np
import pandas as pd

# Number of attributes (columns)
num_attributes = len(data.columns)
print(f"Number of attributes: {num_attributes}")

# Number of patient records (rows)
num_patients = len(data)
print(f"Number of patient records: {num_patients}")

print(data.isnull().sum())



# Handle missing values (example: fill with mean for numerical columns)
for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(data[col].mean())
print(data.isnull().sum())
data.describe()
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print(data.head())
for col in data.columns:
    if pd.api.types.is_bool_dtype(data[col]):
        data[col] = data[col].astype(int)

print(data.head())


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Create the heatmap
plt.figure(figsize=(17, 15))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Identify features highly correlated with the target variable
target_column = 'target'  # Replace 'target' with your actual target column
threshold = 0.3  # Set a threshold for significant correlation
correlated_features = correlation_matrix[target_column][abs(correlation_matrix[target_column]) > threshold].sort_values(ascending=False)

# Exclude the target itself
correlated_features = correlated_features[correlated_features.index != target_column]

print("Highly correlated features with the target:")
print(correlated_features)



target_counts = data['target'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Target Variable')
plt.show()


# prompt: bar graph

import matplotlib.pyplot as plt

# Assuming 'data' DataFrame is already created and processed as in your previous code

# Example: Create a bar chart of the 'age' distribution
plt.figure(figsize=(10, 6))
plt.hist(data['sex_Male'], bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('sex')
plt.ylabel('target')
plt.title('sex vs target')
plt.show()
data.info()


# prompt: svm code

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Import train_test_split

# Assuming 'data' DataFrame is already created and processed as in your previous code
# and 'target' column contains the target variable

# Separate features (X) and target (y)
X = data1.drop(columns=['target'])  # Replace 'target' with the actual target column name
y = data1['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Adjust test_size and random_state as needed

# Initialize the SVM classifier
svm_model = SVC(kernel='linear')  # You can change the kernel (e.g., 'rbf', 'poly')

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")



# prompt: random forest

from sklearn.ensemble import RandomForestClassifier

# Assuming X_train, X_test, y_train, y_test are already defined from previous code

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators

# Train the Random Forest Classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")



# prompt: naive bayes

from sklearn.naive_bayes import GaussianNB

# Assuming X_train, X_test, y_train, y_test are already defined from previous code

# Initialize the Gaussian Naive Bayes classifier
gnb_classifier = GaussianNB()

# Train the Gaussian Naive Bayes classifier
gnb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Gaussian Naive Bayes Model Accuracy: {accuracy * 100:.2f}%")





# prompt: graphical representation of rf and lvq comparism

import matplotlib.pyplot as plt

# Assuming you have the accuracy scores for LVQ and Random Forest
lvq_accuracy = 0.8  # Replace with your actual LVQ accuracy
rf_accuracy = 0.9  # Replace with your actual Random Forest accuracy

# Create a bar chart
models = ['LVQ', 'Random Forest']
accuracies = [lvq_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of LVQ and Random Forest Accuracies')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.show()





# prompt: xgboost, SVM, Random Forest graph represent

import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the accuracy scores for each model stored in a dictionary
model_accuracies = {
    'SVM': 0.85,  # Replace with your actual SVM accuracy
    'Random Forest': 0.92,  # Replace with your actual Random Forest accuracy
    'Gaussian Naive Bayes': 0.78, # Replace with your actual GaussianNB accuracy
    'LVQ': 0.80, # Replace with your actual LVQ accuracy
    'Decision Tree': 0.75, # Replace with your actual Decision Tree accuracy
    'XGBoost': 0.88, # Replace with your actual XGBoost accuracy
    'KNN': 0.82 # Replace with your actual KNN accuracy
}

# Create a DataFrame from the dictionary
results_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])

# Create the bar plot
plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', rot=0)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.7, 1.0) # adjust y-axis limits for better visualization
plt.tight_layout()
plt.show()




# prompt: xgboost, svm, random forest (precision, accuracy, sensitivity, specificity, recall f-measure) plot this in bar graph

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Assuming y_test and y_pred are already defined for each model

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp/(tp+fn)
    return accuracy, precision, recall, f1, specificity, sensitivity


# Evaluate each model
svm_accuracy, svm_precision, svm_recall, svm_f1, svm_specificity, svm_sensitivity = evaluate_model(y_test, svm_model.predict(X_test))
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_specificity, rf_sensitivity = evaluate_model(y_test, rf_classifier.predict(X_test))
xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_specificity, xgb_sensitivity = evaluate_model(y_test, xgb_classifier.predict(X_test))


# Store the results in a dictionary
model_results = {
    'SVM': {'Accuracy': svm_accuracy, 'Precision': svm_precision, 'Recall': svm_recall, 'F1-measure': svm_f1, 'Specificity': svm_specificity, 'Sensitivity': svm_sensitivity},
    'Random Forest': {'Accuracy': rf_accuracy, 'Precision': rf_precision, 'Recall': rf_recall, 'F1-measure': rf_f1, 'Specificity': rf_specificity, 'Sensitivity': rf_sensitivity},
    'XGBoost': {'Accuracy': xgb_accuracy, 'Precision': xgb_precision, 'Recall': xgb_recall, 'F1-measure': xgb_f1, 'Specificity': xgb_specificity, 'Sensitivity': xgb_sensitivity},
}

# Create a DataFrame from the dictionary
results_df = pd.DataFrame(model_results).T

# Plotting the results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-measure', 'Specificity', 'Sensitivity']
plt.figure(figsize=(12, 6))

X_axis = np.arange(len(metrics))

for i, model in enumerate(results_df.index):
  plt.bar(X_axis + (i - 1) * 0.2, results_df.loc[model].values, width=0.2, label = model)

plt.xticks(X_axis, metrics)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.show()



# prompt: roc

from sklearn.metrics import roc_curve, auc

# Assuming you have a trained model (e.g./, svm_model) and X_test, y_test

# Get predicted probabilities for the positive class
y_scores = svm_model.decision_function(X_test) # or predict_proba(X_test)[:, 1] for models with predict_proba

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()