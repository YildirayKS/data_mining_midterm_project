#Importing Required Libraries
#I imported the necessary libraries for data manipulation, scaling, regression modeling, performance evaluation, and visualization.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1. Loading the Dataset
#I loaded the auto MPG dataset from a CSV file
data = pd.read_csv("chapter_3/auto_mpg.csv")

#2. Cleaning the 'horsepower' Column
#I filled the missing values in the 'horsepower' column with the mean
data['horsepower'] = data['horsepower'].replace('?', np.nan).astype(float)
#I filled the missing values in the 'horsepower' column with the mean
data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)
#I dropped the 'car name' column as it was not needed for modeling
if 'car name' in data.columns:
    data.drop(columns=['car name'], inplace=True)

#3. Splitting Features and Target Variable
#I separated the independent variables (features) and the dependent variable (target: mpg)
X = data.drop(columns=['mpg'])
y = data['mpg']

#4. Scaling Features
#I standardized the features to ensure all variables were on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#5. Splitting the Dataset into Training and Testing Sets
#I split the dataset into 70% training and 30% testing data for model evaluation, setting a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=53)

#6. Linear Regression Model
#I created and trained a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
#I made predictions using the trained Linear Regression model
y_pred_lr = lr_model.predict(X_test)

#Performance Metrics (Linear Regression)
#I calculated the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score for the Linear Regression model
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

#7. Random Forest Regressor Model
#I created and trained a Random Forest Regressor model with 100 trees
rf_model = RandomForestRegressor(random_state=53, n_estimators=100)
rf_model.fit(X_train, y_train)
#I made predictions using the trained Random Forest Regressor model
y_pred_rf = rf_model.predict(X_test)

#Performance Metrics (Random Forest)
#I calculated the MSE, MAE, and R² score for the Random Forest model
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

#8. Printing Results
#I displayed the performance metrics for both models
print("Linear Regression Performance:")
print(f"MSE: {lr_mse:.2f}, MAE: {lr_mae:.2f}, R²: {lr_r2:.2f}")
print("\nRandom Forest Performance:")
print(f"MSE: {rf_mse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}")

#9. Visualization
#I plotted the actual vs. predicted values for both models
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.7)
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values (MPG)")
plt.ylabel("Predicted Values")
plt.legend()
plt.get_current_fig_manager().set_window_title("Actual vs Predicted")
plt.show()

#10. Feature Importance (Random Forest)
#I displayed the feature importance scores calculated by the Random Forest model
feature_importances = rf_model.feature_importances_
feature_names = data.drop("mpg", axis=1).columns
print(feature_importances)

#I visualized the feature importance scores
plt.figure(figsize=(7, 5))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.get_current_fig_manager().set_window_title("Feature Importances")
plt.show()