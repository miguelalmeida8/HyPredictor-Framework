from flask import Flask, render_template, request
from collections import deque
import paho.mqtt.client as mqtt
import pandas as pd
import joblib
import psycopg2
import threading
import datetime
import dill
import numpy as np
import lime.lime_tabular
import os
import matplotlib.pyplot as plt
import json

app = Flask(__name__)

# Define global variables
prediction = 0
oil_temperature = 0.00
dv_pressure = 0.00
motor_current = 0.00
tp2 = 0.00
h1 = 0.00
reservoirs = 0.00
towers = 0.00
nr = 21
prediction_probabilities = 0

v_oil_temperature = []
v_dv_pressure = []
v_motor_current = []
v_tp2 = []
v_h1 = []
v_reservoirs = []
v_towers = []
v_predictions = []
v_timestamps = []

# Define the rules list
rules = []

explanation_text = ""

# Rule 1
condition_1 = f"df['median_Oil_temperature'].iloc[-1] > 83"
pred_1 = 0
rules.append((condition_1, pred_1))

# Rule 2
condition_2 = f"df['median_Oil_temperature'].iloc[-1] < 67.25"
pred_2 = 1
rules.append((condition_2, pred_2))

# Rule 3
condition_3 = f"df['median_Oil_temperature'].iloc[-1] > 75.65 and df['median_DV_pressure'].iloc[-1] > -0.02"
pred_3 = 0
rules.append((condition_3, pred_3))



# base de dados
conn = psycopg2.connect(
    dbname='sie2363',
    user='sie2363',
    password='Ola_gato_2023',
    host='db.fe.up.pt',
    port='5432'
)

cur = conn.cursor()

schema_name = 'data'
table_name = 'test_data'

delete_query = f"DELETE FROM {schema_name}.{table_name};"
cur.execute(delete_query)
reset_sequence_query = f"ALTER SEQUENCE {schema_name}.{table_name}_id_seq RESTART WITH 1;"
cur.execute(reset_sequence_query)
conn.commit()

insert_query = """
INSERT INTO {}.test_data (dv_pressure, oil_temperature, motor_current, tp2, h1, reservoirs, towers, failure, timestamp)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
""".format(schema_name)
###################

# Define the MQTT broker's address and port
broker_address = "localhost"  # replace this with your broker's address
broker_port = 1883  # default port for MQTT

# Define the topic you want to subscribe to
topic = "data"

# Data structure to store received data with timestamps
data_window = deque(maxlen=900)  # Assuming 10 seconds * 900 = 15 minutes

# Load your machine learning model
model = joblib.load("C:/Users/migue/PycharmProjects/Hybrid_Approach_Metro_Dataset/Metro_Dataset/Direito/best_model__light.pkl")  # Adjust the file path as per your model location


# Define the path to the explainer file
explainer_file_path = 'models/explainer_lime.pkl'

# Load the explainer object from the file
with open(explainer_file_path, 'rb') as f:
    explainer = dill.load(f)


def save_interactive_visualization(html_path):
    # Generate and save the interactive visualization HTML file
    with open(html_path, 'w') as f:
        f.write('<html><body>Interactive Visualization</body></html>')



def generate_explanation(df):
    global explanation_text, prediction_probabilities

    # Generate Lime explanation
    explanation = explainer.explain_instance(df.values[0], model.predict_proba, num_features=7)

    explanation_text = ""
    for feature, score in explanation.as_list():
        feature = feature.replace("median_", "").replace("_"," ").title()
        explanation_text += f"{feature}: {score}<br>"

    explanation_list = explanation.as_list()
    explanation_dict = dict(explanation_list)

    # Modify Y labels to remove prefix, split at underscores, and capitalize words
    y_labels = [' '.join(word.capitalize() for word in label.split('_')[1:]) for label in explanation_dict.keys()]

    # Plot the explanation with modified Y labels
    plt.figure(figsize=(14, 12))
    plt.barh(y_labels, explanation_dict.values())
    plt.xlabel('Contribution', fontsize=17)
    plt.ylabel('Feature', fontsize=17)
    plt.title(f'Feature Contributions to Prediction (Prediction: {prediction})', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('static/images/explanation_plot.png')

    # Assume explanation is the LIME explanation object
    prediction_probabilities = explanation.predict_proba
    max_index = np.argmax(prediction_probabilities)
    prediction_probabilities = prediction_probabilities[max_index] * 100

# Function to perform feature engineering
def feature_engineering(df):
    df.drop(columns=['Unnamed: 0', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses',
                     'TP3', 'MPG', 'DV_eletric', 'COMP'], inplace=True)
    interval = '15min'
    features = ['DV_pressure', 'Oil_temperature', 'Motor_current', 'TP2', 'H1',
                'Reservoirs', 'Towers']

    for feature in features:
        median_feature = df.set_index('timestamp').resample(interval)[feature].median()
        median_feature_name = f'median_{feature}'  # Name for the new column
        df[median_feature_name] = df['timestamp'].dt.floor(interval).map(
            median_feature)  # Map median values to the DataFrame

    # Drop specified columns
    df.drop(columns=['timestamp', 'DV_pressure', 'Oil_temperature', 'Motor_current', 'TP2', 'H1',
                'Reservoirs', 'Towers'], inplace=True)


    return df


def on_message(client, userdata, message):
    BATCH_SIZE = 10
    global data_window
    global v_oil_temperature, v_dv_pressure, v_motor_current, v_tp2, v_h1, v_reservoirs, v_towers, v_predictions, v_timestamps
    global prediction, timestamps, oil_temperature, dv_pressure, motor_current, tp2, h1, reservoirs, towers
    payload = message.payload.decode()
    print("Received MQTT message:", payload)

    # Split the payload into individual feature values
    feature_values = payload.split(',')

    # Extract the timestamp from the second column
    timestamp = pd.to_datetime(feature_values.pop(1))  # Remove timestamp from feature values list

    # Convert feature values to appropriate data types
    feature_values = [float(value) if value != '' else None for value in feature_values]

    # Append the feature values and timestamp to the data window
    data_window.append((timestamp, feature_values))

    if len(data_window) >= 900:
        # If so, remove the oldest data points
        data_window.popleft()

    df = pd.DataFrame(data_window, columns=['timestamp', 'feature_values'])

    # Expand the feature_values into separate columns
    feature_values_df = pd.DataFrame(df['feature_values'].to_list(),
                                     columns=['Unnamed: 0', 'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                                              'Oil_temperature', 'Motor_current', 'COMP',
                                              'DV_eletric', 'Towers', 'MPG', 'LPS',
                                              'Pressure_switch', 'Oil_level', 'Caudal_impulses'])

    # Concatenate timestamp and expanded feature_values DataFrame
    df = pd.concat([df['timestamp'], feature_values_df], axis=1)

    # Perform feature engineering
    df = feature_engineering(df)

    # Call your prediction function with the pre-processed data
    prediction = model.predict(df)[0]  # Make prediction using loaded model

    ####################RULES#######################

    '''
    # Print each rule in the rules list
    for i, (condition, pred) in enumerate(rules, 1):
        print(f"Rule {i}:")
        print(f"Condition: {condition}")
        print(f"Prediction: {pred}")
        print()
    '''

    apply_rules(df)


    ################################################

    v_oil_temperature.append(df['median_Oil_temperature'].iloc[-1])
    v_dv_pressure.append(df['median_DV_pressure'].iloc[-1])
    v_motor_current.append(df['median_Motor_current'].iloc[-1])
    v_tp2.append(df['median_TP2'].iloc[-1])
    v_h1.append(df['median_H1'].iloc[-1])
    v_reservoirs.append(df['median_Reservoirs'].iloc[-1])
    v_towers.append(df['median_Towers'].iloc[-1])
    v_predictions.append(int(prediction))
    v_timestamps.append(timestamp)

    oil_temperature = df['median_Oil_temperature'].iloc[-1]
    dv_pressure = df['median_DV_pressure'].iloc[-1]
    motor_current = df['median_Motor_current'].iloc[-1]
    tp2 = df['median_TP2'].iloc[-1]
    h1 = df['median_H1'].iloc[-1]
    reservoirs = df['median_Reservoirs'].iloc[-1]
    towers = df['median_Towers'].iloc[-1]
    prediction = int(prediction)

    # Generate LIME explanation
    generate_explanation(df)

    print(len(data_window))
    # Check if it's time to commit
    if len(data_window) % BATCH_SIZE == 0:
        # Reopen the connection
        conn = psycopg2.connect(
            dbname='sie2363',
            user='sie2363',
            password='Ola_gato_2023',
            host='db.fe.up.pt',
            port='5432'
        )
        # Open a cursor
        cur = conn.cursor()

        # Insert data into the database
        for i in range(BATCH_SIZE):
            cur.execute(insert_query, (
            v_dv_pressure[i], v_oil_temperature[i],v_motor_current[i], v_tp2[i], v_h1[i], v_reservoirs[i], v_towers[i],
            v_predictions[i], v_timestamps[i]))

        # Commit changes
        conn.commit()

        # Close the cursor
        cur.close()
        conn.close()

        # Clear the lists
        v_oil_temperature.clear()
        v_dv_pressure.clear()
        v_motor_current.clear()
        v_tp2.clear()
        v_h1.clear()
        v_reservoirs.clear()
        v_towers.clear()
        v_predictions.clear()
        v_timestamps.clear()




def mqtt_loop():
    # Create an MQTT client instance
    client = mqtt.Client()

    # Set callback function for message reception
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(broker_address, broker_port)

    # Subscribe to the topic
    client.subscribe(topic)

    # Start the MQTT client loop to receive messages
    client.loop_forever()

# Start MQTT client loop in a separate thread
mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.start()

@app.route('/')
def index():

    return render_template('front_page.html', prediction=prediction, oil_temperature=oil_temperature,
                           dv_pressure=dv_pressure, motor_current=motor_current, TP2=tp2, towers=towers, h1=h1)


@app.route('/add_failure_report')
def add_failure_report():
    # Logic for the Add Failure Report page
    return render_template('page_failure_report.html')


@app.route('/submit_form', methods=['POST'])
def submit_form():
    global nr, model
    if request.method == 'POST':
        try:
            conn = psycopg2.connect(
                dbname='sie2363',
                user='sie2363',
                password='Ola_gato_2023',
                host='db.fe.up.pt',
                port='5432'
            )
            cur = conn.cursor()
            # Parse form data
            start_date = datetime.datetime.strptime(request.form['start_date'], '%Y-%m-%dT%H:%M')
            end_date = datetime.datetime.strptime(request.form['end_date'], '%Y-%m-%dT%H:%M')
            type_of_failure = request.form['type_of_failure']
            severity = request.form['severity']

            # Begin transaction
            cur.execute("BEGIN")

            # Insert failure report into the database
            insert_query = """
            INSERT INTO {}.failure_report (nr, start_time, end_time, failure, severity)
            VALUES (%s, %s, %s, %s, %s);
            """.format(schema_name)

            nr += 1
            cur.execute(insert_query, (nr, start_date, end_date, type_of_failure, severity))

            # Update test data based on conditions
            update_query = """
            UPDATE {}.test_data
            SET failure = 
                CASE
                    WHEN timestamp BETWEEN %s AND %s THEN 1
                    WHEN timestamp < %s THEN 0
                    ELSE failure
                END;
            """.format(schema_name)

            cur.execute(update_query, (start_date, end_date, start_date))

            # Commit transaction
            conn.commit()

            # Retrain the model with the updated data
            query = """
                SELECT * 
                FROM {}.test_data
                WHERE timestamp < %s
            """.format(schema_name)

            # Fetch updated dataset from the database
            df = pd.read_sql(query, conn, params=[end_date])
            conn.close()

            # Drop unnecessary columns
            df.drop(columns=['id', 'timestamp'], inplace=True)

            # Prepare features and target variable
            X = df.drop(columns=['failure'])  # Features
            y = df['failure']  # Target variable

            # Retrain the model
            model.fit(X, y)


            # Redirect to a success page or render a success message
            return render_template('page_failure_report.html')

        except Exception as e:
            # Rollback transaction on error
            conn.rollback()
            # Handle exception (e.g., log error, display error message)
            return str(e)


@app.route('/submit_rules', methods=['POST'])
def submit_rules():
    global rules
    # Get user-defined rules from the form
    variable = request.form['word']
    operator = request.form['operator']
    value = float(request.form['value'])
    pred = int(request.form['pred'])

    # Construct the condition based on user input
    condition = f"df['{variable}'].iloc[-1] {operator} {value}"

    # Append the condition and prediction as a tuple to the rules list
    rules.append((condition, pred))
    print(condition)
    print(pred)

    return render_template('page_add_rules.html')


@app.route('/submit_rules_2', methods=['POST'])
def submit_rules_2():
    global rules
    # Get user-defined rules from the form
    variable = request.form['word']
    operator = request.form['operator']
    value = float(request.form['value'])
    variable_2 = request.form['word_2']
    operator_2 = request.form['operator_2']
    value_2 = float(request.form['value_2'])
    pred = int(request.form['pred'])

    # Construct the condition based on user input
    condition = f"df['{variable}'].iloc[-1] {operator} {value} and df['{variable_2}'].iloc[-1] {operator_2} {value_2}"

    # Append the condition and prediction as a tuple to the rules list
    rules.append((condition, pred))
    print(condition)
    print(pred)

    return render_template('page_add_rules.html')


def apply_rules(df):
    global prediction, rules

    for condition, pred in rules:
        # Evaluate the condition dynamically
        if prediction == pred and eval(condition):
            # Update prediction if condition is met
            if pred == 0:
                prediction = 1
            else:
                prediction = 0
            break  # Exit loop after first rule match (assuming rules are ordered by priority)


@app.route('/open_page_add_rule')
def open_page_add_rule():

    return render_template('page_add_rules.html')

@app.route('/open_page_add_rule_2', methods=['GET', 'POST'])
def open_page_add_rule_2():
    print("ola")
    return render_template('page_add_rules_2.html')



@app.route('/show_rules')
def show_rules():
    global rules

    formatted_conditions = []

    # Iterate over the rules
    for condition, pred in rules:
        # Split the condition by 'and' to handle multiple conditions
        conditions = condition.split(" and ")

        # Initialize a list to store the formatted conditions
        formatted_condition_parts = []

        # Iterate over each condition part
        for cond in conditions:
            # Extract variable, operator, and value from the condition
            variable_parts = cond.split("['")[1].split("']")[0].split("_")
            variable = " ".join(variable_parts[1:]).capitalize()  # Join parts except the first one and capitalize

            operator = cond.split(" ")[-2]
            value = cond.split(" ")[-1]

            # Format the condition part
            formatted_condition_part = f"{variable} {operator} {value}"

            # Append the formatted condition part to the list
            formatted_condition_parts.append(formatted_condition_part)

        # Join the formatted condition parts with 'and' to reconstruct the original condition
        formatted_condition = " and ".join(formatted_condition_parts)

        # Append the formatted condition and prediction as a tuple to the new vector
        formatted_conditions.append((formatted_condition, pred))

        # Print the formatted condition
        print(formatted_condition)

    indexed_rules = [(index, rule) for index, rule in enumerate(formatted_conditions)]
    return render_template('page_show_rules.html', indexed_rules=indexed_rules)


@app.route('/delete_rule', methods=['POST'])
def delete_rule():
    global rules
    index = int(request.form['index'])

    # Delete the rule by index
    del rules[index]

    return show_rules()



@app.route('/update_rule_order', methods=['POST'])
def update_rule_order():

    global rules

    new_order = request.form.get('rule-order')  # Get the new order from the form
    new_order = json.loads(new_order)

    new_order = [int(index.split('_')[1]) for index in new_order]
    new_order = [int(index) for index in new_order]
    rules = [rules[index] for index in new_order]

    return show_rules()

@app.route('/open_page_xai')
def open_page_xai():
    explanation_list = [pair.split(': ') for pair in explanation_text.split('<br>') if pair.strip()]

    return render_template('page_xai.html', explanation_list=explanation_list, narrative_explanation=explanation_text, prediction = prediction, prediction_probabilities = prediction_probabilities)

if __name__ == '__main__':
    app.run(debug=False)
