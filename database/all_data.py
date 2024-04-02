import psycopg2


def create_all_data_table(db_host, db_user, db_password, db_name, existing_table_name, all_data_table_name, schema=None):
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        dbname=db_name
    )
    cursor = conn.cursor()

    if schema:
        cursor.execute(f"SET search_path TO {schema}")

    # Step 1: Create the new table "all_data" with an additional column "failure"
    create_table_query = f"""
    CREATE TABLE {all_data_table_name} AS
    SELECT *, 0 AS failure
    FROM {existing_table_name};
    """
    cursor.execute(create_table_query)

    # Step 2: Update the "failure" column based on conditions
    update_query = f"""
    UPDATE {all_data_table_name}
    SET failure = 1
    WHERE EXISTS (
        SELECT 1
        FROM failure_report
        WHERE {all_data_table_name}.timestamp BETWEEN failure_report."start_time" AND failure_report."end_time"
    );
    """
    cursor.execute(update_query)

    # Commit changes and close connection
    conn.commit()
    conn.close()


# Database details

csv_file = r'C:\Users\migue\Desktop\metropt_dataset\MetroPT3(AirCompressor).csv'
db_host = 'db.fe.up.pt'
db_user = 'sie2363'
db_password = 'Ola_gato_2023'
db_name = 'sie2363'
existing_table_name = 'data'
all_data_table_name = 'all_data'
schema = 'data'

# Call the function to create the "all_data" table and update the "failure" column
create_all_data_table(db_host, db_user, db_password, db_name, existing_table_name, all_data_table_name, schema)
