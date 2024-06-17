import csv
import time
import paho.mqtt.client as mqtt

# Define the MQTT broker's address and port
broker_address = "localhost"  # replace this with your broker's address
broker_port = 1883  # default port for MQTT

# Define the topic you want to publish to
topic = "data"  # replace this with the topic you want to publish to


# Function to publish data from CSV file

def publish_data(filename):
    # Create an MQTT client instance
    client = mqtt.Client()

    # Connect to the MQTT broker
    client.connect(broker_address, broker_port)

    # Read data from CSV file and publish
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            message = ",".join(row)  # Assuming each row is a list of values
            client.publish(topic, message)
            print("Published:", message)
            time.sleep(10)  # Delay for 10 seconds before publishing the next line

    # Disconnect from the broker
    client.disconnect()

# Replace 'your_dataset.csv' with the path to your CSV file
publish_data("C:/Users/migue/Downloads/test_data.csv")
