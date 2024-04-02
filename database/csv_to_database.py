import psycopg2

def create_table(csv_file, db_host, db_user, db_password, db_name, table_name, schema=None):
    print("Connecting to the PostgreSQL database...")
    conn = psycopg2.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        dbname=db_name
    )
    print("Connection established.")

    cursor = conn.cursor()

    # If a schema is provided, set the search path
    if schema:
        cursor.execute(f"SET search_path TO {schema}")

    # Create the table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            "Unnamed: 0" FLOAT,
            timestamp TIMESTAMP,
            TP2 FLOAT,
            TP3 FLOAT,
            H1 FLOAT,
            DV_pressure FLOAT,
            Reservoirs FLOAT,
            Oil_temperature FLOAT,
            Motor_current FLOAT,
            COMP FLOAT,
            DV_eletric FLOAT,
            Towers FLOAT,
            MPG FLOAT,
            LPS FLOAT,
            Pressure_switch FLOAT,
            Oil_level FLOAT,
            Caudal_impulses FLOAT
        )
    """)

    print("Copying data from CSV file to the database...")
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        next(csvfile)
        cursor.copy_from(csvfile, table_name, sep=',', null='')  # Assuming CSV format with comma separator

    conn.commit()
    conn.close()
    print("Table creation complete.")

# Example usage
csv_file = r'C:\Users\migue\Desktop\metropt_dataset\MetroPT3(AirCompressor).csv'
db_host = 'db.fe.up.pt'
db_user = 'sie2363'
db_password = 'Ola_gato_2023'
db_name = 'sie2363'
table_name = 'data'
schema = 'data'  # Specify the schema here if needed

create_table(csv_file, db_host, db_user, db_password, db_name, table_name, schema)
