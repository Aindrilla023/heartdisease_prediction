import numpy as np
import pandas as pd

df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')
print(df.head())
# prompt: catagorical to numerical

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Iterate through each column in the DataFrame
for col in df.columns:
    # Check if the column is of object type (categorical)
    if df[col].dtype == 'object':
        # Fit and transform the column using LabelEncoder
        df[col] = le.fit_transform(df[col])

# Print the updated DataFrame
print(df.head())

# prompt: SNR best feature collection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming df is already loaded from the previous code block
# If not, uncomment the following lines:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')


# Separate features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
feature_importance_df


# prompt: PCC best feature collection

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (assuming it's already uploaded as 'HeartDiseaseTrain-Test.csv')
try:
    df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')
except FileNotFoundError:
    print("Error: 'HeartDiseaseTrain-Test.csv' not found. Please upload the file.")
    from google.colab import files
    uploaded = files.upload()
    df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')




# Preprocess the data (handle categorical features)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Prepare the data for modeling
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (or any other suitable model)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# Feature importance analysis
feature_importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top N most important features (e.g., top 13)
top_n = 13  # Change this value to display a different number of features
print(f"\nTop {top_n} most important features:")
print(feature_importance_df.head(top_n))




# prompt: collect the features from PCC and SNR

# Assuming 'feature_importance_df' is already created from the previous code
# Get the indices of the top N features based on importance
top_n_indices = feature_importance_df.index[:10]

# Get the names of the top N features
top_n_features = feature_importance_df['Feature'][top_n_indices]

print("\nTop features:")
top_n_features


# prompt: svm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, y_test are already defined from previous code

# Initialize and train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)  # You can experiment with different kernels
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")



# prompt: LVQ

import numpy as np

class LVQ:
    def __init__(self, n_prototypes, learning_rate, epochs):
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.prototypes = None
        self.prototype_labels = None

    def initialize_prototypes(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.prototypes = np.zeros((self.n_prototypes * n_classes, n_features))
        self.prototype_labels = np.zeros(self.n_prototypes * n_classes)

        for i in range(n_classes):
            class_indices = np.where(y == i)[0]
            random_indices = np.random.choice(class_indices, size=self.n_prototypes, replace=False)
            self.prototypes[i * self.n_prototypes:(i + 1) * self.n_prototypes, :] = X[random_indices, :]
            self.prototype_labels[i * self.n_prototypes:(i + 1) * self.n_prototypes] = i

    def fit(self, X, y):
        self.initialize_prototypes(X, y)
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                distances = np.linalg.norm(self.prototypes - X[i], axis=1)
                winner_index = np.argmin(distances)
                if y[i] == self.prototype_labels[winner_index]:
                    self.prototypes[winner_index, :] += self.learning_rate * (X[i] - self.prototypes[winner_index, :])
                else:
                    self.prototypes[winner_index, :] -= self.learning_rate * (X[i] - self.prototypes[winner_index, :])
        print("Finished training LVQ.")

    def predict(self, X):
      predictions = []
      for i in range(X.shape[0]):
        distances = np.linalg.norm(self.prototypes - X[i], axis=1)
        winner_index = np.argmin(distances)
        predictions.append(int(self.prototype_labels[winner_index]))
      return np.array(predictions)

# Example usage with your existing code:
# Assuming X_train, X_test, y_train, y_test are already defined from your previous code

# Initialize and train the LVQ model
lvq_classifier = LVQ(n_prototypes=2, learning_rate=0.1, epochs=100)  # Adjust parameters as needed
lvq_classifier.fit(X_train.values, y_train.values)  # Fit the model

# Make predictions on the test set
y_pred_lvq = lvq_classifier.predict(X_test.values)

# Evaluate the model's accuracy
accuracy_lvq = accuracy_score(y_test, y_pred_lvq)
print(f"LVQ Accuracy: {accuracy_lvq}")



# prompt: random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, y_test are already defined from previous code

# Initialize and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# prompt: random forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
try:
    df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')
except FileNotFoundError:
    print("Error: 'HeartDiseaseTrain-Test.csv' not found. Please upload the file.")
    from google.colab import files
    uploaded = files.upload()
    df = pd.read_csv('/content/HeartDiseaseTrain-Test.csv')

# Preprocess the data (handle categorical features)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Prepare the data for modeling
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")



# prompt: KNN

from sklearn.neighbors import KNeighborsClassifier

# Assuming X_train, X_test, y_train, y_test are already defined from previous code

# Initialize and train a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn_classifier.predict(X_test)

# Evaluate the model's accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn}")



# prompt: performance graph

import matplotlib.pyplot as plt

# Assuming accuracy_rf, accuracy_svm, accuracy_lvq, accuracy_logreg, accuracy_knn are defined
# from previous code blocks.  Replace with actual variable names if different.

models = ['Random Forest', 'SVM', 'LVQ', 'Logistic Regression', 'KNN']
accuracies = [accuracy, accuracy_svm, accuracy_lvq, accuracy_logreg, accuracy_knn]  # Replace with your accuracy values

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# prompt: model performence according to fi , recall, accuracy, specivity, sensetivity and graph

from sklearn.metrics import classification_report, confusion_matrix

# Assuming y_test and y_pred are already defined from previous code blocks.

# Generate the classification report
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate metrics from the confusion matrix
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
recall = sensitivity # recall is the same as sensitivity

print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming cm is the confusion matrix from the previous code block

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()




# prompt: xgboost, svm, random forest (precision, accuracy, sensitivity, specificity, recall f-measure) plot this in bar graph

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC # Importing the SVM classifier
from xgboost import XGBClassifier # Importing the XGBoost classifier

!pip install xgboost

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

# Initialize and fit the SVM model here before evaluation:
svm_model = SVC(kernel='linear', random_state=42) # You can experiment with different kernels
svm_model.fit(X_train, y_train)

# Initialize and fit the XGBoost model here before evaluation:
xgb_classifier = XGBClassifier(random_state=42) # You can adjust hyperparameters here
xgb_classifier.fit(X_train, y_train)

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