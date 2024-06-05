# Графики 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from dataSet import*
import functionML
"""
# 1. Ящичковая диаграмма для зависимости распределения цены от марки авто 
functionML.get_boxplot(df, 'Model', 'Price') # полный датасет 
functionML.get_boxplot(train, 'Model', 'Price') # train (изначальный)
# 2. Ящичковая диаграмма для тестовой выборки 20 самых распространенных моделей
functionML.get_boxplot(train[train['Model'].
                           isin(train['Model'].value_counts()[:20].index.tolist())], 
                           'Model', 'Price')
"""
# 3. Столбчатая диаграмма усеченного датасета Train и его логарифма 
functionML.Train_graph()

# 4. Диаграммы категориальных данных 
train['bodyType'].value_counts(ascending=False).plot(kind='barh',figsize=(7,6))
plt.show()

train['Year'].value_counts(ascending=False).plot(kind='barh',figsize=(8,15))
plt.show()

train['Color'].value_counts(ascending=False).plot(kind='barh',figsize=(8,8))
plt.show()

train['HorsePower'].value_counts(ascending=False).plot(kind='barh',figsize=(8,15))
plt.show()

train['FuelType'].value_counts(ascending=False).plot(kind='barh',figsize=(8,3))
plt.show()

train['numberOfDoors'].value_counts(ascending=False).plot(kind='barh',figsize=(8,3))
plt.show()

train['Transmission'].value_counts(ascending=False).plot(kind='barh',figsize=(8,3))
plt.show()

train['Engine Capacity(L)'].value_counts(ascending=False).plot(kind='barh',figsize=(8,3))
plt.show()

# песочиница для признаков 
correlations = data.corrwith(data['Price']).sort_values(ascending=False)
fig, ax = plt.subplots(figsize = (6, 10))
plot = sns.barplot(y=correlations.index, x=correlations).set(xlabel='Correlation Index')
plt.show()


