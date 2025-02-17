import streamlit as st
from sklearn.calibration import LabelEncoder
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from web.utils import merge_dataframes
from scipy import stats
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_connection():
    return psycopg2.connect(
        user='postgres',
        password='123Secret_a',
        host='localhost',
        port='5432',
        database='loldb'
    )

@st.cache_data
def load_data():
    logging.info("Загрузка данных из базы данных.")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    tables = cursor.fetchall()
    
    dataframes = {}
    for table in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{table[0]}"', conn)
        dataframes[table[0]] = df
        logging.info(f"Загружена таблица: {table[0]}")
    
    merged = merge_dataframes(dataframes, [x[0] for x in tables])
    for item in merged:
        dataframes[item["key"]] = item["dataframe"]
    
    cursor.close()
    conn.close()
    return dataframes

def prepare_dataframe(df):
    logging.info("Подготовка датафрейма.")
    statistics = {}
    original_columns = list(df.columns)
    
    for col in original_columns:
        # Convert to numeric if possible
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            logging.warning(f"Не удалось конвертировать столбец {col}: {e}")
            pass

        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_info = {
                'type': 'numeric',
                'missing_ratio': df[col].isnull().mean(),
                'max': df[col].max(),
                'min': df[col].min(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'variance': df[col].var(),
                'quantile_0_1': df[col].quantile(0.1),
                'quantile_0_9': df[col].quantile(0.9),
                'quartile_1': df[col].quantile(0.25),
                'quartile_3': df[col].quantile(0.75)
            }
            
            # Fill missing values
            if stats_info['missing_ratio'] > 0:
                df[col] = df[col].fillna(stats_info['mean'])
            
            statistics[col] = stats_info

        # Handle categorical columns
        else:
            le = LabelEncoder()
            new_col = f'{col}_Code'
            df[new_col] = le.fit_transform(df[col])
            df.drop(col, axis=1, inplace=True)
            
            stats_info = {
                'type': 'categorical',
                'missing_ratio': df[new_col].isnull().mean(),
                'unique_count': df[new_col].nunique(),
                'mode': df[new_col].mode()[0] if not df[new_col].empty else None
            }
            statistics[new_col] = stats_info
            
    return statistics

# Загрузка данных
dataframes = load_data()

# Отображение гипотез в сайдбаре
st.sidebar.header("Проверка гипотез")
if 'stats1' in dataframes:
    df_stats = dataframes['stats1'].copy()
    prepare_dataframe(df_stats)
    
    try:
        group1 = df_stats[df_stats['win'] == 1]['goldearned']
        group2 = df_stats[df_stats['win'] == 0]['goldearned']
        t, p = stats.ttest_ind(group1, group2, equal_var=False)
        st.sidebar.markdown(f"**Гипотеза 1:** Золото (p-value = {p:.4f})")
        
        group1 = df_stats[df_stats['win'] == 1]['kills']
        group2 = df_stats[df_stats['win'] == 0]['kills']
        t, p = stats.ttest_ind(group1, group2)
        st.sidebar.markdown(f"**Гипотеза 2:** Убийства (p-value = {p:.4f})")
    except KeyError as e:
        logging.error(f"Ошибка в данных: {e}")
        st.sidebar.error(f"Ошибка в данных: {e}")

# Основной интерфейс
st.title("Анализ данных LoL")
selected_table = st.selectbox("Выберите таблицу", options=list(dataframes.keys()))
st.session_state.selected_table = selected_table

if selected_table:
    df = dataframes[selected_table].copy()
    stats = prepare_dataframe(df)

    # Отображение статистики
    st.header("Статистика столбцов")
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    st.dataframe(stats_df)
    
    # Тепловая карта корреляций
    st.header("Корреляционная матрица")
    numeric_cols = [col for col, info in stats.items() if info['type'] == 'numeric']
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr[corr.abs() > 0.5], annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Недостаточно числовых столбцов для построения корреляций")

    # Pairplot
    st.header("Парные графики")
    
    # Состояние для хранения выбранных столбцов
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []

    selected_columns = st.multiselect(
        "Выберите столбцы для визуализации",
        options=list(stats.keys()),
        default=st.session_state.selected_columns
    )
    
    # Обновление состояния выбранных столбцов
    st.session_state.selected_columns = selected_columns
    
    if selected_columns:
        fig = sns.pairplot(df[selected_columns], diag_kind='kde')
        st.pyplot(fig.fig)
