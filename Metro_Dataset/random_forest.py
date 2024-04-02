import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.preprocessing import StandardScaler

DATA = r'C:\Users\migue\Desktop\metropt_dataset\MetroPT3(AirCompressor).csv'

data = pd.read_csv(DATA, na_values="na")


import pandas as pd

# Creating the failure report DataFrame
failure_report_data = {
    'Nr.': [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'Start Time': [
    '4/12/2020 11:50', '4/18/2020 00:00', '4/19/2020 00:00', '4/29/2020 03:20', '4/29/2020 22:00', '5/13/2020 14:00',
    '5/18/2020 05:00', '5/19/2020 10:10', '5/19/2020 22:10', '5/20/2020 00:00', '5/23/2020 09:50', '5/29/2020 23:30',
    '5/30/2020 00:00', '6/01/2020 15:00', '6/03/2020 10:00', '6/05/2020 10:00', '6/06/2020 00:00', '6/07/2020 00:00',
    '7/08/2020 17:30', '7/15/2020 14:30', '7/17/2020 04:30'],

    'End Time': [
    '4/12/2020 23:30', '4/18/2020 23:59', '4/19/2020 01:30', '4/29/2020 04:00', '4/29/2020 22:20', '5/13/2020 23:59',
    '5/18/2020 05:30', '5/19/2020 11:00', '5/19/2020 23:59', '5/20/2020 20:00', '5/23/2020 10:10', '5/29/2020 23:59',
    '5/30/2020 06:00', '6/01/2020 15:40', '6/03/2020 11:00', '6/05/2020 23:59', '6/06/2020 23:59', '6/07/2020 14:30',
    '7/08/2020 19:00', '7/15/2020 19:00', '7/17/2020 05:30'],

    'Failure': ['Air leak', 'Air Leak', 'Air Leak', 'Air Leak', 'Air leak', 'Air Leak', 'Air Leak', 'Air Leak', 'Air leak', 'Air Leak', 'Air Leak', 'Air Leak', 'Air leak', 'Air Leak', 'Air Leak', 'Air Leak', 'Air leak', 'Air Leak', 'Air Leak', 'Air Leak', 'Air Leak'],
    'Severity': ['High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress', 'High stress'],
}

failure_report = pd.DataFrame(failure_report_data)

# Convert "Start Time" and "End Time" columns to datetime format
failure_report['Start Time'] = pd.to_datetime(failure_report['Start Time'])
failure_report['End Time'] = pd.to_datetime(failure_report['End Time'])

failure_report['Start Time'] -= pd.Timedelta(hours=2)

# Display the failure report DataFrame
print(failure_report)

# Initialize a new column "failure" in train_data with default value 0
data['failure'] = 0

data['timestamp'] = pd.to_datetime(data['timestamp'])

# Iterate over each row in the failure report
for index, row in failure_report.iterrows():
    start_time = row['Start Time']
    end_time = row['End Time']
    # Find rows in train_data where the "timestamp" falls within the range
    mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
    # Update the "failure" column to 1 for matching rows
    data.loc[mask, 'failure'] = 1


################################  Feature Enginnering #####################################

# Resample the DataFrame to 10-minute intervals and calculate median of 'DV_pressure'
median_dv_pressure = data.set_index('timestamp').resample('10min')['DV_pressure'].median()

# Merge the calculated median values back to the original DataFrame
data = data.merge(median_dv_pressure, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True, suffixes=('', '_median'))

# Rename the new column
data.rename(columns={'DV_pressure_median': 'median_dv_pressure'}, inplace=True)

# Drop unnecessary columns
data.drop(columns=['key_0'], inplace=True)


# Resample the DataFrame to 10-minute intervals and calculate median of 'Oil_temperature'
median_oil_temperature = data.set_index('timestamp').resample('10min')['Oil_temperature'].median()

# Merge the calculated median values back to the original DataFrame
data = data.merge(median_oil_temperature, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True)

# Rename the new column
data.rename(columns={'Oil_temperature_y': 'median_oil_temperature'}, inplace=True)

data.drop(columns=['key_0'], inplace=True)



# Resample the DataFrame to 10-minute intervals and calculate median of 'Motor_current'
median_motor_current = data.set_index('timestamp').resample('10min')['Motor_current'].median()


# Merge the calculated median values back to the original DataFrame
data = data.merge(median_motor_current, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True)

# Rename the new column
data.rename(columns={'Motor_current_y': 'median_motor_current'}, inplace=True)

data.drop(columns=['key_0'], inplace=True)


# Resample the DataFrame to 10-minute intervals and calculate median of 'TP3'
median_tp3 = data.set_index('timestamp').resample('10min')['TP3'].median()

# Merge the calculated median values back to the original DataFrame
data = data.merge(median_tp3, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True)

# Rename the new column
data.rename(columns={'TP3_y': 'median_tp3'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)


# Resample the DataFrame to 10-minute intervals and calculate median of 'H1'
median_h1 = data.set_index('timestamp').resample('10min')['H1'].median()

# Merge the calculated median values back to the original DataFrame
data = data.merge(median_h1, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True)

# Rename the new column
data.rename(columns={'H1_y': 'median_h1'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)


# Resample the DataFrame to 10-minute intervals and calculate median of 'Towers'
median_towers = data.set_index('timestamp').resample('10min')['Towers'].median()

# Merge the calculated median values back to the original DataFrame
data = data.merge(median_towers, how='left', left_on=data['timestamp'].dt.floor('10min'), right_index=True)

# Rename the new column
data.rename(columns={'Towers_y': 'median_towers'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)


print(data.columns)

###########################################################################################################

# Display the updated train_data DataFrame
#print(data)

filtered_data = data[data["timestamp"].dt.date == pd.to_datetime('2020-05-29').date()]

#print(filtered_data)


data = data.drop(columns=['Reservoirs'])
data = data.drop(columns=['Unnamed: 0'])
data = data.drop(columns=['COMP'])
data = data.drop(columns=['TP2'])
data = data.drop(columns=['Oil_temperature_x'])
#data = data.drop(columns=['DV_pressure_x'])
data = data.drop(columns=['MPG'])
data = data.drop(columns=['DV_eletric'])
data = data.drop(columns=['Motor_current_x'])
data = data.drop(columns=['TP3_x'])
data = data.drop(columns=['H1_x'])
data = data.drop(columns=['Towers_x'])

# Given date to split the dataset
given_date = pd.to_datetime('2020-06-04')  # June 4, 2020

# Filter out rows before April 2020
data_filter = data[data['timestamp'] >= '2020-04-12']
#data_filter = data
#data_filter = data_filter[data_filter['timestamp'] <= '2020-07-05']

# Filter data based on the given date
train_data = data_filter[data_filter['timestamp'] < given_date].drop(columns=['timestamp'])
test_data = data_filter[data_filter['timestamp'] >= given_date].drop(columns=['timestamp'])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

print("Training dataset size:", len(train_data))
print("Test dataset size:    ", len(test_data))

# Split features and target variables for training and testing
x_train = train_data.drop(columns=['failure'])  # Features for training set
y_train = train_data['failure']  # Target variable for training set

x_test = test_data.drop(columns=['failure'])  # Features for test set
y_test = test_data['failure']  # Target variable for test set




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
import joblib

# Use the best parameters obtained from grid search
# best_params = search.best_params_

# best_clf = RandomForestClassifier(n_estimators=best_params['clf__n_estimators'],
#                                    max_depth=best_params['clf__max_depth'],
#                                    class_weight="balanced")

scoring = make_scorer(recall_score)

best_clf = RandomForestClassifier(n_estimators=500,
                                   max_depth=20,
                                   class_weight="balanced")

# Define the parameter grid
param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [None, 20, 30, 40],
}

# Create the Random Forest classifier
rf_classifier = RandomForestClassifier(class_weight='balanced')

# Create the GridSearchCV object
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring=scoring, verbose=4)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best estimator (model)
best_model = grid_search.best_estimator_

# Get the best parameters
best_params = grid_search.best_params_

print("Best parameters found during grid search:")
print(grid_search.best_params_)


# Predictions on the training set
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

# Calculate evaluation metrics on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1_score = f1_score(y_train, y_train_pred)

# Calculate evaluation metrics on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the evaluation metrics
print("\nTraining Set Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1_score)
print("\nTest Set Metrics:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1_score)


# Create confusion matrix
conf_mat = confusion_matrix(y_test, y_test_pred)

# Define class labels
class_labels = ["False", "Pos"]

# Plot confusion matrix with custom labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

#################    RULES    ###########################333

test_predictions_with_rules = pd.Series(y_test_pred)
test_predictions_with_rules.index = y_test.index

count=0
certa=0

for n in range(len(test_predictions_with_rules)):
    if test_predictions_with_rules[n] == 0 and x_test.iloc[n]['median_oil_temperature'] > 83:
        test_predictions_with_rules[n] = 1
        count = count+1
        if y_test[n] == 1:
            certa = certa+1

print(count)
print(certa)

count=0
certa=0

for n in range(len(test_predictions_with_rules)):
    if test_predictions_with_rules[n] == 1 and x_test.iloc[n]['median_oil_temperature'] < 67 and x_test.iloc[n]['Oil_level'] == 0:

        test_predictions_with_rules[n] = 0
        count = count+1
        if y_test[n] == 0:
            certa = certa+1

print(count)
print(certa)

count=0
certa=0

for n in range(len(test_predictions_with_rules)):
    if (test_predictions_with_rules[n] == 1 and x_test.iloc[n]['median_oil_temperature'] < 67.25):

        test_predictions_with_rules[n] = 0
        count = count+1
        if y_test[n] == 0:
            certa = certa+1

print(count)
print(certa)

##############################################



# Calculate evaluation metrics with expert rules applied
test_accuracy_e = accuracy_score(y_test, test_predictions_with_rules)
test_precision_e = precision_score(y_test, test_predictions_with_rules)
test_recall_e = recall_score(y_test, test_predictions_with_rules)
test_f1_score_e = f1_score(y_test, test_predictions_with_rules)

# Print the evaluation metrics
print("\nTest Accuracy (with expert rules):", test_accuracy_e)
print("Test Precision (with expert rules):", test_precision_e)
print("Test Recall (with expert rules):", test_recall_e)
print("Test F1 Score (with expert rules):", test_f1_score_e)

print("\nRecall Difference:", (test_recall_e-test_recall)*100)
print("Precision Difference:", (test_precision_e-test_precision)*100)
print("F1-score Difference:", (test_f1_score_e-test_f1_score)*100)
print("Accuracy Difference:", (test_accuracy_e-test_accuracy)*100)

# Create confusion matrix
conf_mat = confusion_matrix(y_test, test_predictions_with_rules)

# Define class labels
class_labels = ["False", "Pos"]

# Plot confusion matrix with custom labels
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

if test_recall > 0 and test_precision > 0:
    print("\nSalvou o Modelo")
    joblib.dump(best_model, 'best_model_rf.pkl')