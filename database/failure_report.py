import psycopg2
import pandas as pd

def insert_failure_report_to_db(failure_report_df, db_host, db_user, db_password, db_name, table_name, schema=None):
    print("Connecting to the PostgreSQL database...")
    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        dbname=db_name
    )
    print("Connection established.")

    cursor = conn.cursor()

    if schema:
        cursor.execute(f"SET search_path TO {schema}")

    # Create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Nr INT,
            Start_Time TIMESTAMP,
            End_Time TIMESTAMP,
            Failure TEXT,
            Severity TEXT
        )
    """)

    print("Inserting failure report data into the database...")
    for index, row in failure_report_df.iterrows():
        cursor.execute(f"""
            INSERT INTO {table_name} (Nr, Start_Time, End_Time, Failure, Severity)
            VALUES (%s, %s, %s, %s, %s)
        """, (row['Nr.'], row['Start Time'], row['End Time'], row['Failure'], row['Severity']))

    conn.commit()
    conn.close()
    print("Data insertion complete.")

# Example usage
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

failure_report_df = pd.DataFrame(failure_report_data)

# Convert "Start Time" and "End Time" columns to datetime format
failure_report_df['Start Time'] = pd.to_datetime(failure_report_df['Start Time'])
failure_report_df['End Time'] = pd.to_datetime(failure_report_df['End Time'])

# Adjust for timezone if needed
failure_report_df['Start Time'] -= pd.Timedelta(hours=1)

# Database details
db_host = 'db.fe.up.pt'
db_user = 'sie2363'
db_password = 'Ola_gato_2023'
db_name = 'sie2363'
table_name = 'failure_report'
schema = 'data'

# Insert failure report data into the database
insert_failure_report_to_db(failure_report_df, db_host, db_user, db_password, db_name, table_name, schema)
