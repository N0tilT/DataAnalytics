import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Загрузка данных
df = sns.load_dataset('mpg')
print(df.info())

rows, columns = df.shape
print(f'Количество строк: {rows}, Количество столбцов: {columns}')

# Получаем числовые переменные
num_vars = df.select_dtypes(include=['float64', 'int64'])

# Вычисляем необходимые статистики
for column in num_vars.columns:
    missing_ratio = num_vars[column].isnull().sum()
    max_val = num_vars[column].max()
    min_val = num_vars[column].min()
    mean_val = num_vars[column].mean()
    median_val = num_vars[column].median()
    variance = num_vars[column].var()
    quantile_0_1 = num_vars[column].quantile(0.1)
    quantile_0_9 = num_vars[column].quantile(0.9)
    quartile_1 = num_vars[column].quantile(0.25)
    quartile_3 = num_vars[column].quantile(0.75)

    print(f'Для {column}:')
    print(f'  Доля пропусков: {missing_ratio:.2f}')
    print(f'  Максимальное: {max_val}, Минимальное: {min_val}')
    print(f'  Среднее: {mean_val}, Медиана: {median_val}')
    print(f'  Дисперсия: {variance}, Квантиль 0.1: {quantile_0_1}, Квантиль 0.9: {quantile_0_9}')
    print(f'  Квартиль 1: {quartile_1}, Квартиль 3: {quartile_3}')

    if missing_ratio>0:
        print("Заменим пустые значения на среднее арифметическое")    
        df[column].fillna(df[column].mean(), inplace = True)
        print(f"Доля пропусков для {column}:{num_vars[column].isnull().sum()}")

# Получаем категориальные переменные
cat_vars = df.select_dtypes(include=['object'])

le = LabelEncoder()

for column in cat_vars.columns:
    missing_ratio = cat_vars[column].isnull().mean()
    unique_count = cat_vars[column].nunique()
    mode = cat_vars[column].mode()[0]

    print(f'Для {column}:')
    print(f'  Доля пропусков: {missing_ratio:.2f}')
    print(f'  Количество уникальных значений: {unique_count}')
    print(f'  Мода: {mode}')

    # Применяем Label Encoding
    df[f'{column}_Code'] = le.fit_transform(df[column])
    df.drop(column,axis=1,inplace=True)


# Гипотеза 1: Средний расход топлива (mpg) у 4 и 6 цилиндров
mpg_4_cylinders = df[df['cylinders'] == 4]['mpg']
mpg_6_cylinders = df[df['cylinders'] == 6]['mpg']

t_stat, p_value = stats.ttest_ind(mpg_4_cylinders, mpg_6_cylinders)
print("Гипотеза 1. Соотношение среднего расхода у 4 и 6 цилиндров:") 
print("t-статистика =", t_stat, ", p-значение =", p_value)
print("Гипотеза верна" if p_value>0.05 else "Гипотеза не верна")

# Гипотеза 2: Проверка зависимости mpg от weight (коэффициент корреляции)
correlation, p_value = stats.pearsonr(df['mpg'], df['weight'])
print("Гипотеза 2. Проверка зависимости mpg от weight:")
print( "Коэффициент корреляции =", correlation, ", p-значение =", p_value)
print("Гипотеза верна" if p_value>0.05 else "Гипотеза не верна")

# Гипотеза 3: Проверка зависимости mpg от horsepower (корреляция)
correlation, p_value = stats.pearsonr(df['mpg'], df['horsepower'])
print("Гипотеза 3. Проверка зависимости mpg от horsepower:")
print("Коэффициент корреляции =", correlation, ", p-значение =", p_value)
print("Гипотеза верна" if p_value>0.05 else "Гипотеза не верна")


# Гипотеза 4: ANOVA тест для mpg в зависимости от origin_Code
f_stat, p_value = stats.f_oneway(df[df['origin_Code'] == 0]['mpg'], 
                                  df[df['origin_Code'] == 1]['mpg'], 
                                  df[df['origin_Code'] == 2]['mpg'])
print("Гипотеза 4. Влияние страны производителя на расход топлива:")
print("F-статистика =", f_stat, ", p-значение =", p_value)
print("Гипотеза верна" if p_value>0.05 else "Гипотеза не верна")

correlation_matrix = df.corr()
print(correlation_matrix)


import matplotlib.pyplot as plt
# Визуализация матрицы корреляции
plt.figure(figsize=(12, 10))  # Установка размеров фигуры
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Матрица корреляции', fontsize=18)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig("correlation.png")  # Отображение матрицы корреляции


# Определяем целевую переменную и признаки
y = df['mpg']
X_horsepower = df[['horsepower']]
X_weight = df[['weight']]
X_horsepower = (X_horsepower - X_horsepower.mean()) / X_horsepower.std()
X_weight = (X_weight - X_weight.mean()) / X_weight.std()

# Функция для обычного градиентного спуска
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros((X.shape[1] + 1, 1))  # добавляем 1 для b (свободный член)
    X_b = np.c_[np.ones((m, 1)), X]  # добавляем x0 = 1 к каждому экземпляру

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y.values.reshape(-1, 1))
        theta -= learning_rate * gradients

    return theta

# Функция для стохастического градиентного спуска
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=1000):
    m = len(y)
    theta = np.zeros((X.shape[1] + 1, 1))  # добавляем 1 для b
    X_b = np.c_[np.ones((m, 1)), X]

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index + 1]  # случайный экземпляр
            yi = y.iloc[random_index:random_index + 1].values.reshape(-1, 1)
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients

    return theta

# Применение методов
theta_gd_horsepower = gradient_descent(X_horsepower, y)
theta_gd_weight = gradient_descent(X_weight, y)

theta_sgd_horsepower = stochastic_gradient_descent(X_horsepower, y)
theta_sgd_weight = stochastic_gradient_descent(X_weight, y)

print("Обычный градиентный спуск (horsepower):", theta_gd_horsepower)
print("Обычный градиентный спуск (weight):", theta_gd_weight)
print("Стохастический градиентный спуск (horsepower):", theta_sgd_horsepower)
print("Стохастический градиентный спуск (weight):", theta_sgd_weight)