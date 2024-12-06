#Importing Required Libraries
#I imported the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#1. Loading the Dataset
#I loaded the dataset and read it
dataset_path = "chapter_1/diabetes.csv"
diabetes_data = pd.read_csv(dataset_path)

#2. Preparing the Data
#I replaced zero values with the median of the respective columns in specific features
columns_to_fill = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_fill:
    diabetes_data[col] = diabetes_data[col].replace(0, np.nan)
    diabetes_data[col].fillna(diabetes_data[col].median(), inplace=True)

#Splitting features and target
#I separated the independent variables (features) and the dependent variable (target)
features = diabetes_data.drop(columns=["Outcome"])
target = diabetes_data["Outcome"]

#3. Dividing the Data Set into Training and Testing Sets
#I split the dataset into 70% training and 30% testing data for model evaluation, ensuring reproducibility of results by setting a random state
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=53)

#4. Training Decision Tree Classifier
#I created and trained a Decision Tree Classifier model
decision_tree_classifier = DecisionTreeClassifier(random_state=53)
decision_tree_classifier.fit(X_train, y_train)

#Predicting using the Decision Tree model
#I made predictions on the test set using the trained model
dt_predictions = decision_tree_classifier.predict(X_test)

#Calculating performance metrics for Decision Tree
#I measured accuracy, precision, recall, and F1 score for the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

#Visualizing Confusion Matrix for Decision Tree
#I displayed the confusion matrix to analyze the Decision Tree model's performance
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(dt_conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#5. Training Random Forest Classifier
#I created and trained a Random Forest Classifier model
random_forest_classifier = RandomForestClassifier(random_state=53)
random_forest_classifier.fit(X_train, y_train)

#Predicting using the Random Forest model
#I made predictions on the test set using the trained model
rf_predictions = random_forest_classifier.predict(X_test)

#Calculating performance metrics for Random Forest
#I measured accuracy, precision, recall, and F1 score for the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

#Visualizing Confusion Matrix for Random Forest
#I displayed the confusion matrix to analyze the Random Forest model's performance
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(rf_conf_matrix, annot=True, cmap="Greens", fmt="d")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#6. Training Support Vector Machine Classifier (SVM)
#I created and trained an SVM Classifier model
svm_classifier = SVC(random_state=53)
svm_classifier.fit(X_train, y_train)

#Predicting using the SVM model
#I made predictions on the test set using the trained SVM model
svm_predictions = svm_classifier.predict(X_test)

#Calculating performance metrics for SVM
#I measured accuracy, precision, recall, and F1 score for the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)

#Visualizing Confusion Matrix for SVM
#I displayed the confusion matrix to analyze the SVM model's performance
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(svm_conf_matrix, annot=True, cmap="Reds", fmt="d")
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#7. Summarizing Results
#I created a summary table with performance metrics, including accuracy, precision, recall, and F1 score, for all the models
results_df = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "SVM"],
    "Accuracy": [dt_accuracy, rf_accuracy, svm_accuracy],
    "Precision": [dt_precision, rf_precision, svm_precision],
    "Recall": [dt_recall, rf_recall, svm_recall],
    "F1 Score": [dt_f1, rf_f1, svm_f1],
})

#I displayed the results table to compare the performance metrics of all the models
print(results_df)