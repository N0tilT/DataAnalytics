import os
import pandas as pd
import psycopg2
from psycopg2 import sql

dbname = 'loldb'
user = 'postgres'
password = '123Secret_a'
host = 'localhost'
port = '5432'

conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
cursor = conn.cursor()
conn.autocommit = True

try:
    cursor.execute(f"CREATE DATABASE {dbname};")
except psycopg2.errors.DuplicateDatabase:
    print(f"База данных {dbname} уже существует.")

csv_folder = './data'

# Проход по всем файлам .csv в папке
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        splitted = []
        path = os.path.join(csv_folder, filename)
        with open(path,"r") as table:
            splitted = table.read().split('\n')

        with open(path,"w") as table:
            table.write(str.join("\n",[i.strip(',') for i in splitted]))

        table_name = filename[:-4]

        df = pd.read_csv(os.path.join(csv_folder, filename))

        columns = [col for col in df.columns.tolist() if col != 'id']
        columns_with_types = ", ".join([f"{col} VARCHAR" for col in columns])

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {columns_with_types}
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()

        for _, row in df.iterrows():
            row_data = row[row.index != 'id'] 
            insert_query = sql.SQL(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})")
            cursor.execute(insert_query, tuple(row_data))

        conn.commit()
        print(f"Загружены данные в таблицу {table_name}")

selected_table = 'champs'
cursor.execute(f"SELECT * FROM {selected_table};")
rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.close()
conn.close()
