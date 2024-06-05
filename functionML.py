# универсальные функции
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations
#from dataSet import trainPlt

def mape(y_true, y_pred):
    '''
    Метрика
    '''
    return np.mean(np.abs((y_pred-y_true)/y_true))

def del_columns(df, list_):
    '''
    "Тихое" удаление колонок датафрейма по списку 
    '''
    for i in list_:
        try: 
            df.drop([i],axis=1, inplace=True)
        except:
            continue
    return 

def display_data(dataframe, list_columns=None):
    '''
    Развернутая информация по датасету с фильтрацией по списку колонок,
    по умолчнию без фильтрации,
    названия признаков упорядочены по алфавиту
    '''
    if list_columns==None:
        index_ = sorted(dataframe.columns)
    else: 
        index_ = sorted([x for x in set(dataframe.dtypes.index).intersection(set(list_columns))])
    
    df = dataframe[index_]
    d = pd.concat([df.dtypes,df.count() + df.isna().sum(),\
               round((df.isna().sum()/(df.count() + df.isna().sum()))*100,2),df.nunique(),],axis=1)
    d.columns = ['Тип', 'Общ.кол', '% пропусков','Кол-во уник.значений']
    display(d)
    return

def compare_dataframes(df1, df2, show_=0):
    '''
    Находит соответствие названий колонок двух датафреймов,
    возвращает колонки объединенного датафрейма и общие колонки:
    похволяет проверить готовность датафреймов к слиянию.
    Показывает таблицу соответствия колонок в соответствии со значением
    
    show_:
        0 - не показывать
        1 - показывать только различия
        2 - показывать все       
   
    
    '''
    # список всех колонок
    total_index = sorted(list(set(df1.columns.tolist()) | set(df2.columns.tolist())))
    common_index = sorted(list(set(df1.columns.tolist()).intersection(set(df2.columns.tolist()))))
    if show_ > 0:
        d = pd.concat([pd.DataFrame(total_index, index = total_index),\
                   pd.DataFrame(df1[df1.columns.tolist()].dtypes, index = df1.columns),\
                   pd.DataFrame(df2[df2.columns.tolist()].dtypes, index = df2.columns)],axis=1)

        d.columns = ['Названия признаков', f'Датасет 1', 'Датасет 2']
        
        
        if show_ == 2:
            display(d[['Датасет 1','Датасет 2']])
        else:
            
            display(d[d['Датасет 1']!=d['Датасет 2']][['Датасет 1','Датасет 2']])
            
    return total_index, common_index

def IQR_outlier(df: pd.DataFrame, column: object, verbose: bool=True) -> tuple:
    '''
    Функция для отображения границ межквартильного размаха
    '''
    perc25 = round(df[column].quantile(0.25),3)
    perc75 = round(df[column].quantile(0.75),3)
    IQR = perc75 - perc25
    low = perc25 - 1.5*IQR
    high = perc75 + 1.5*IQR
    if verbose:
        print(column)
        print('25-й перцентиль: {},'.format(perc25)[:-1], '75-й перцентиль: {},'.format(perc75),
            "IQR: {}, ".format(IQR), "Границы выбросов: [{f}, {l}].".format(f=low, l=high))
    return (low, high)

def cut_outlier(df, col):
    '''
    Функция на основании статистики
    для числового признака создает новый числовой признак
    с "убиранными хвостами" 
    '''
    low, high = IQR_outlier(df,col, False) # границ межквартильного размаха
    new_col = col+'_c'
    df[new_col] = df[col]
    df[new_col] = df[new_col].apply(lambda x: x if x>low else low)
    df[new_col] = df[new_col].apply(lambda x: x if x<high else high)
    
    return

def cut_tails(df, col, quality=10):
    '''
    Функция на основании заданного количества 
    для категориального признака создает новый признак
    с заданным количеством значений данного признака,
    остальным значениям присваивается 'OTHERS'
    '''
    new_col = col+'_c'
    df[new_col] = df[col]
    _index = df[new_col].value_counts().index[:quality] # сохраняем индексы
    # Новое значение если текст - OTHERS, число и пр. - порядковый номер последнего элемента
    new_value = 'OTHERS'
    df[new_col] = df[new_col].apply(lambda x: x if x in _index else new_value)
    
    return

def get_boxplot(df, column, _target):
    '''
    plot boxes для колонки
    '''
    fig, ax = plt.subplots(figsize = (20, 4))
    sns.boxplot(x=column, y=_target,data=df, ax=ax) 
    plt.xticks(rotation=90,fontsize=8)
    ax.set_title('Boxplot for ' + column)
    plt.show()
    return

def show_correlation(df, features):
    '''
    plot correlation matrix
    '''
    
    #corr_matrix = df.drop(categorial_feature, axis=1).corr()
    corr_matrix = df[features].corr()
    plt.rcParams['figure.figsize'] = (15,15)
    sns.heatmap(corr_matrix, square=True,
                annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu");
    plt.tight_layout()
    return

def get_stat_dif(df: pd.DataFrame, column: object, _target='score') -> bool:
    '''
    Функция для проведения теста Стьюдента для номинативных и смешанных переменных
    '''
    cols = df.loc[:, column].value_counts()
    cols = cols[cols>15].index
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], _target].dropna(), 
                        df.loc[df.loc[:, column] == comb[1], _target].dropna()).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони #0.075
            print('Найдены статистически значимые различия для колонки', column)
            return True
            break
    return 

def get_stat_dif(df: pd.DataFrame, column: object, _target='score') -> bool:
    '''
    Функция для проведения теста Стьюдента для номинативных и смешанных переменных
    '''
    cols = df.loc[:, column].value_counts()
    cols = cols[cols>15].index
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], _target].dropna(), 
                        df.loc[df.loc[:, column] == comb[1], _target].dropna()).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони #0.075
            print('Найдены статистически значимые различия для колонки', column)
            return True
            break
    return 

def fullfil_omissions(df, _features, arr_grp = [['brand','model','name','modelDate','productionDate'],\
                                                ['brand','model','name','modelDate'],\
                                                ['brand','model','name'],\
                                                ['brand','model'],\
                                                ['brand'],\
                                                []], _type=0):

    '''
    _features - список свойств

    _type - тип признаков:
    0 - числовые
    1 - категориальные

    arr_grp - список группировок
    По умолчанию:
    arr_grp = [['brand','model_name'],['brand'],[]]

    Залатывает пропуски,
    сначала по группе: бренд+модель,
    потом - бренд
    затем в целом по датасету

    '''
    
    # Таблица  данных с посчитанными пропусками
    tabl = df[_features].isnull().sum()

    for el in arr_grp:
        
        if len(el)>0:
            grp = df.groupby(el)

            # цикл по проускам в числовых признаках для замены средним по группе
            for ind in list(tabl[tabl>0].index):
                
                try:
                    if _type == 0:
                        
                        df[ind] = np.round(df[ind].fillna(grp[ind].transform('mean')))
                    else:
                        #value_ = grp[ind].value_counts().index[0][len(el)] # самое распросстраненное значение в группе
                        value_ = grp[ind].median()
                        df[ind].fillna(value_, inplace=True)

                except Exception as err:
                    
                    print(err, ind)
                    
                    if _type == 1: # не сдаемся
                        value_ = grp[ind].value_counts().index[0][len(el)]  # самое распросстраненное значение в группе
                        print(el, ind, value_)
                        df[ind].fillna(value_, inplace=True)
                        
                    continue

        else:

            # цикл по проускам в числовых признаках для замены оставшихся пропусков
            for ind in list(tabl[tabl>0].index):
                                 
                if _type == 0:
                                 
                    df[ind] = np.round(df[ind].interpolate(method='polynomial', order=2))
                else:
                                 
                    df[ind].fillna(df[ind].mode()[0], inplace=True)
    return

def create_year_month_from_start_date():
    '''
    Для всех датасетов подготавливает поля из колонки start_date
    
    Доработать до универсального инструмента!
    
    '''
    for df in ['train','train_upd','test']:
        if df == 'test':
            y = f"{df}[\'start_date\'] = pd.to_datetime({df}[\'parsing_unixtime\'],unit=\'s\')"
        else:  
            print(df)
            y = f"{df}[\'start_date\'] = pd.to_datetime({df}[\'start_date\'])"
        print(y)    
        exec(y)    

        y = f"{df}[\'sale_year\'] = {df}[\'start_date\'].dt.year"
        exec(y)

        y = f"{df}[\'sale_month\'] = {df}[\'start_date\'].dt.month"

        exec(y)

        y = f"del_columns({df}, delete_columns)"

        exec(y)
    
    return 

def show_weights(features, weights, scales):
    '''
    Визуализация весов соответствующих признаков в обучаемом датасете 
    с учетом регуляризации (параметр scales)
    '''
    fig, axs = plt.subplots(figsize=(14, 10), ncols=2)
    sorted_weights = sorted(zip(weights, features, scales), reverse=True)
    weights = [x[0] for x in sorted_weights]
    features = [x[1] for x in sorted_weights]
    scales = [x[2] for x in sorted_weights]
    sns.barplot(y=features, x=weights, ax=axs[0])
    axs[0].set_xlabel("Weight")
    sns.barplot(y=features, x=scales, ax=axs[1])
    axs[1].set_xlabel("Scale")
    plt.tight_layout()

    return

"""
def Train_graph():  
    #1. График зависимости цены авто от их количества 
    sns.set_style()
    fig, ax = plt.subplots(figsize = (8, 4))
    sns.histplot(trainPlt['Price'], bins=70).set_title\
    ("Распределение целевой функции датасета TRAIN \n после выбора ценового сегмента \n 200 тыс. - 7 млн. руб");
    plt.xticks(np.arange(min(trainPlt['Price']), max(trainPlt['Price'])+300000, 300000.0))
    plt.show()
    # 2. Распределение логарифма
    sns.histplot(np.log(1+trainPlt['Price'])).set(title="Распределение логарифма целевой функции",\
                                           xlabel='log(Price)');
    plt.show()
    return()
"""