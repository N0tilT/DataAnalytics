from flask import Flask, render_template, request
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import merge_dataframes


def create_app():
    app = Flask(__name__)
    def get_connection():
        connection = psycopg2.connect(
            user = 'postgres',
            password = '123Secret_a',
            host = 'postgres',
            port = '5432',
            database = 'loldb'
        )
        return connection

    dataframes = {}
    def init_app():
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = cursor.fetchall()

        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM {table[0]}', conn)
            dataframes[table[0]] = df
        merged = merge_dataframes(dataframes,[x[0] for x in tables])
        for item in merged:
            dataframes[item["key"]] = item["dataframe"]
        cursor.close()
        conn.close()

    init_app()

    @app.route('/')
    def index():
        tables = list(dataframes.keys())
        return render_template('index.html', tables=tables)

    @app.route('/select_columns', methods=['POST'])
    def select_columns():
        table_name = request.form['table']
        df = dataframes[table_name]
        
        numeric = {str(x):False for x in df.columns}
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
                numeric[column]=True
            except ValueError:
                numeric[column]=False
        heatmap = ""
        if not any(numeric.values()):
            return render_template('select_columns.html', table=table_name, columns=not_ids, heatmap=heatmap)
        ids = []
        not_ids = []
        for x in df.columns:
            if "id" not in x and numeric[x]:
                not_ids.append(x)
            else:
                ids.append(x)
        if(len(not_ids)>0):
            plt.figure(figsize=(12, 8))
            corr_matrix = df[not_ids].corr()
            threshold = 0.5

            filtered_corr_matrix = corr_matrix[corr_matrix.abs() > threshold].dropna(how='all', axis=0).dropna(how='all', axis=1)
            sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Тепловая карта корреляции')
            plt.savefig('./web/static/correlation_heatmap.png')
            heatmap = './static/correlation_heatmap.png'
            plt.close()
        
        return render_template('select_columns.html', table=table_name, not_ids=not_ids, ids=ids, heatmap=heatmap)

    @app.route('/plot', methods=['POST'])
    def plot():
        table_name = request.form['table']
        selected_columns = request.form.getlist('columns')
        df = dataframes[table_name][selected_columns]
        sns.pairplot(df, diag_kind='kde')
        plt.savefig('./web/static/plot.png')
        plt.close()
        
        return render_template('plot.html', plot='./static/plot.png', table=table_name)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
