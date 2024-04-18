import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

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
#Interval
interval = '15min'

#DV_pressure
median_dv_pressure = data.set_index('timestamp').resample(interval)['DV_pressure'].median()
data = data.merge(median_dv_pressure, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True, suffixes=('', '_median'))
data.rename(columns={'DV_pressure_median': 'median_dv_pressure'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['DV_pressure'])

#Oil_temperature
median_oil_temperature = data.set_index('timestamp').resample(interval)['Oil_temperature'].median()
data = data.merge(median_oil_temperature, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Oil_temperature_y': 'median_oil_temperature'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Oil_temperature_x'])

#Motor_current
median_motor_current = data.set_index('timestamp').resample(interval)['Motor_current'].median()
data = data.merge(median_motor_current, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Motor_current_y': 'median_motor_current'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Motor_current_x'])

#TP3
median_tp3 = data.set_index('timestamp').resample(interval)['TP3'].median()
data = data.merge(median_tp3, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'TP3_y': 'median_tp3'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['TP3_x'])

#TP2
median_tp2 = data.set_index('timestamp').resample(interval)['TP2'].median()
data = data.merge(median_tp2, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'TP2_y': 'median_tp2'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['TP2_x'])

#H1
median_h1 = data.set_index('timestamp').resample(interval)['H1'].median()
data = data.merge(median_h1, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'H1_y': 'median_h1'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['H1_x'])

#Reservoirs
median_reservoirs = data.set_index('timestamp').resample(interval)['Reservoirs'].median()
data = data.merge(median_reservoirs, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Reservoirs_y': 'median_reservoirs'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Reservoirs_x'])

#Towers
median_towers = data.set_index('timestamp').resample(interval)['Towers'].median()
data = data.merge(median_towers, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Towers_y': 'median_towers'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Towers_x'])

#COMP
median_comp = data.set_index('timestamp').resample(interval)['COMP'].median()
data = data.merge(median_comp, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'COMP_y': 'median_comp'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['COMP_x'])

#DV_electric
median_dv_eletric = data.set_index('timestamp').resample(interval)['DV_eletric'].median()
data = data.merge(median_dv_eletric, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'DV_eletric_y': 'median_dv_eletric'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['DV_eletric_x'])

#MPG
median_mpg = data.set_index('timestamp').resample(interval)['MPG'].median()
data = data.merge(median_mpg, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'MPG_y': 'median_mpg'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['MPG_x'])

#LPS
median_lps = data.set_index('timestamp').resample(interval)['LPS'].median()
data = data.merge(median_lps, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'LPS_y': 'median_lps'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['LPS_x'])

#Pressure_switch
median_pressure_switch = data.set_index('timestamp').resample(interval)['Pressure_switch'].median()
data = data.merge(median_pressure_switch, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Pressure_switch_y': 'median_pressure_switch'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Pressure_switch_x'])

#Oil_level
median_oil_level = data.set_index('timestamp').resample(interval)['Oil_level'].median()
data = data.merge(median_oil_level, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Oil_level_y': 'median_oil_level'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Oil_level_x'])

#Caudal_impulses
median_caudal_impulses = data.set_index('timestamp').resample(interval)['Caudal_impulses'].median()
data = data.merge(median_caudal_impulses, how='left', left_on=data['timestamp'].dt.floor(interval), right_index=True)
data.rename(columns={'Caudal_impulses_y': 'median_caudal_impulses'}, inplace=True)
data.drop(columns=['key_0'], inplace=True)
data = data.drop(columns=['Caudal_impulses_x'])


print(data.columns)

###########################################################################################################
target_variable = 'failure'

# Calculate correlation coefficients between features and target variable
correlation_matrix = abs(data.corr()[target_variable])

# Order correlation coefficients by absolute values
sorted_correlation = correlation_matrix.abs().sort_values(ascending=False)

# Print correlation coefficients for all features
print("Correlation Coefficients with Target Variable (Ordered by Absolute Values):")
for feature in sorted_correlation.index:
    correlation = correlation_matrix[feature]
    print(f"{feature}: {correlation}")

correlation_threshold = 0.2
columns_to_drop = correlation_matrix[correlation_matrix.abs() < correlation_threshold].index
columns_to_drop = [col for col in columns_to_drop if col != 'timestamp']

# Drop the columns from the dataset
data = data.drop(columns=columns_to_drop)

print("Columns Dropped:")
print(columns_to_drop)

correlation_matrix = data.drop(columns=['failure']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)

# Add title and labels
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

high_correlation_pairs = (correlation_matrix.abs() > 0.8) & (correlation_matrix.abs() < 1)

# Get the indices where the condition is True
high_correlation_indices = np.where(high_correlation_pairs)

# Extract the pairs of variables with high correlation
high_correlation_variables = [(correlation_matrix.index[i], correlation_matrix.columns[j]) for i, j in zip(*high_correlation_indices)]

# Print the pairs of variables with high correlation
print("Pairs of Variables with Correlation > 0.8:")
for pair in high_correlation_variables:
    print(pair)

columns_to_drop = ['median_tp3', 'median_mpg', 'median_dv_eletric', 'median_comp']

for column in columns_to_drop:
    if column in data.columns:
        data = data.drop(columns=[column])

######################################################################

filtered_data = data[data["timestamp"].dt.date == pd.to_datetime('2020-05-29').date()]

# Given date to split the dataset
given_date = pd.to_datetime('2020-06-04')  # June 4, 2020

# Filter out rows before April 2020
data_filter = data[data['timestamp'] >= '2020-04-12']
# data_filter = data
# data_filter = data_filter[data_filter['timestamp'] <= '2020-07-05']

# Filter data based on the given date
train_data = data_filter[data_filter['timestamp'] < given_date].drop(columns=['timestamp'])
test_data = data_filter[data_filter['timestamp'] >= given_date].drop(columns=['timestamp'])

data = data.drop(columns=['timestamp'])

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

print("Training dataset size:", len(train_data))
print("Test dataset size:    ", len(test_data))

# Split features and target variables for training and testing
x_train = train_data.drop(columns=['failure'])  # Features for training set
y_train = train_data['failure']  # Target variable for training set

x_test = test_data.drop(columns=['failure'])  # Features for test set
y_test = test_data['failure']  # Target variable for test set

scoring = make_scorer(recall_score)

print(x_train.columns)

param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [1],
    'min_child_samples': [1],
    'reg_lambda': [0.1]
}

'''
param_grid = {
    'learning_rate': [0.01, 0.012, 0.008],
    'n_estimators': [70, 100, 130, 160, 200],
    'max_depth': [1],
    'min_child_samples': [1],
    'reg_lambda': [0.1, 0.01, 0.12, 0.14, 0.008],
}
'''
# Create LightGBM Classifier
model = lgb.LGBMClassifier(verbose=0)

# Create Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring, verbose=5)

# Fit the grid search to the training data
grid_search.fit(x_train, y_train)

# Get the best estimator (model)
best_model = grid_search.best_estimator_

print("Best parameters found during grid search:")
print(grid_search.best_params_)

# Get the best parameters
best_params = grid_search.best_params_

best_model.fit(x_train, np.ravel(y_train))

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

# Find the index of the first failure prediction in the test data
failure_index = None
for idx, pred in enumerate(y_test_pred):
    if pred == 1:
        failure_index = idx
        break

# Print the index of the first failure prediction
print("Index of the first failure prediction in the test data:", failure_index)




import lime
import lime.lime_tabular

# Initialize the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=x_train.values,
                                                   mode='classification',
                                                   feature_names=x_train.columns.tolist())

import dill

# Save the explainer object to a file using dill
with open('explainer_lime.pkl', 'wb') as f:
    dill.dump(explainer, f)


# Select a specific instance from your test dataset for explanation
#instance_index = 26610  # You can choose any index from your test dataset
instance_index = 293851
instance = x_test.iloc[[instance_index]].values[0]

# Get the prediction for the selected instance
prediction = best_model.predict_proba(instance.reshape(1, -1))[0]

# Explain the prediction using LIME
explanation = explainer.explain_instance(instance, best_model.predict_proba, num_features=len(x_train.columns))

# Visualize the explanation
explanation.show_in_notebook()

# Save the explanation to an HTML file
#explanation.save_to_file('explanation.templates')

# Open the HTML file in a web browser
#import webbrowser
#webbrowser.open('explanation.templates')



# Get the explanation in a format that can be plotted
explanation_list = explanation.as_list()

# Convert the explanation list to a dictionary for plotting
explanation_dict = dict(explanation_list)

# Plot the explanation
plt.figure(figsize=(14, 12))
plt.barh(list(explanation_dict.keys()), explanation_dict.values())
plt.xlabel('Contribution')
plt.ylabel('Feature')
plt.title('Feature Contributions to Prediction')
plt.show()



####

# Get the explanation list
explanation_list = explanation.as_list()

# Print the explanation list
print("Rules applied by LIME:")
for feature, weight in explanation_list:
    print(f"Feature: {feature}, Weight: {weight}")



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
    if (test_predictions_with_rules[n] == 1 and x_test.iloc[n]['median_oil_temperature'] < 67.25):

        test_predictions_with_rules[n] = 0
        count = count+1
        if y_test[n] == 0:
            certa = certa+1

print(count)
print(certa)




count=0
certa=0

for n in range(len(test_predictions_with_rules)):
    if (test_predictions_with_rules[n] == 0 and x_test.iloc[n]['median_oil_temperature'] > 75.65 and x_test.iloc[n]['median_dv_pressure'] > -0.02):

        test_predictions_with_rules[n] = 1
        count = count+1
        if y_test[n] == 1:
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

'''
if test_recall > 0 and test_precision > 0:
    print("\nSalvou o Modelo")
    joblib.dump(best_model, 'best_model__light.pkl')
    '''