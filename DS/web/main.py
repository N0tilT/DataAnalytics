from sklearn.calibration import LabelEncoder
from flask import Flask, render_template, request
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import merge_dataframes
from scipy import stats

def create_app():
    app = Flask(__name__)
    def get_connection():
        connection = psycopg2.connect(
            user = 'postgres',
            password = '123Secret_a',
            host = 'localhost',
            port = '5432',
            database = 'loldb'
        )
        return connection
    def prepare_dataframe(df:pd.DataFrame):
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column])
            except:
                pass
            
        # Нормализация данных
        num_vars = df.select_dtypes(include=['float64', 'int64'])
        cat_vars = df.select_dtypes(include=['object'])

        statistics = {}
        
        # Обработка числовых переменных
        for column in num_vars.columns:
            missing_ratio = num_vars[column].isnull().mean()
            max_val = num_vars[column].max()
            min_val = num_vars[column].min()
            mean_val = num_vars[column].mean()
            median_val = num_vars[column].median()
            variance = num_vars[column].var()
            quantile_0_1 = num_vars[column].quantile(0.1)
            quantile_0_9 = num_vars[column].quantile(0.9)
            quartile_1 = num_vars[column].quantile(0.25)
            quartile_3 = num_vars[column].quantile(0.75)

            # Заполнение пропусков
            if missing_ratio > 0:
                df[column] = df[column].fillna(mean_val)

            statistics[column] = {
                'missing_ratio': missing_ratio.item(),
                'max': max_val.item(),
                'min': min_val.item(),
                'mean': mean_val.item(),
                'median': median_val.item(),
                'variance': variance.item(),
                'quantile_0_1': quantile_0_1.item(),
                'quantile_0_9': quantile_0_9.item(),
                'quartile_1': quartile_1.item(),
                'quartile_3': quartile_3.item()
            }
        
        # Обработка категориальных переменных
        le = LabelEncoder()
        for column in cat_vars.columns:
            missing_ratio = cat_vars[column].isnull().mean()
            unique_count = cat_vars[column].nunique()
            mode = cat_vars[column].mode()[0]

            # Применяем Label Encoding
            df[f'{column}_Code'] = le.fit_transform(df[column])
            df.drop(column, axis=1, inplace=True)

            statistics[f'{column}_Code'] = {
                'missing_ratio': missing_ratio.item(),
                'unique_count': unique_count,
                'mode': mode
            }
        return statistics
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

        df = dataframes['stats1']
        print("Рассмотрим таблицу stats1")
        prepare_dataframe(df)
        print(df.info())
        # Гипотеза 1        
        group1_g = df[df['win'] == 1]['goldearned']
        group2_g = df[df['win'] == 0]['goldearned']
        t_stat, p_value = stats.ttest_ind(group1_g, group2_g, equal_var=False)
        print(f'Гипотеза 1. Выигрывающая команда зарабатывает больше золота, чем проигравшая: t-stat = {t_stat}, p-value = {p_value}')
        # Гипотеза 2
        win_kills = df[df['win'] == 1]['kills']
        lose_kills = df[df['win'] == 0]['kills']
        t_stat, p_value = stats.ttest_ind(win_kills, lose_kills)
        print(f'Гипотеза 2. Выигрывающая команда имеет больше убйиств, чем проигравшая: t-stat = {t_stat}, p-value = {p_value}')

    init_app()

    @app.route('/')
    def index():
        tables = list(dataframes.keys())
        return render_template('index.html', tables=tables)

    @app.route('/select_columns', methods=['POST'])
    def select_columns():
        table_name = request.form['table']
        df = dataframes[table_name]
        statistics = prepare_dataframe(df)        
        if len(statistics) > 0:
            plt.figure(figsize=(12, 8))
            corr_matrix = df[list(statistics.keys())].corr()
            threshold = 0.5

            filtered_corr_matrix = corr_matrix[corr_matrix.abs() > threshold].dropna(how='all', axis=0).dropna(how='all', axis=1)
            sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Тепловая карта корреляции')
            plt.savefig('./static/correlation_heatmap.png')
            heatmap = './static/correlation_heatmap.png'
            plt.close()

        return render_template('select_columns.html', table=table_name, not_ids=statistics.keys(), heatmap=heatmap, statistics=statistics)


    @app.route('/plot', methods=['POST'])
    def plot():
        table_name = request.form['table']
        selected_columns = request.form.getlist('columns')
        df = dataframes[table_name][selected_columns]
        sns.pairplot(df, diag_kind='kde')
        plt.savefig('./static/plot.png')
        plt.close()
        
        return render_template('plot.html', plot='./static/plot.png', table=table_name)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
