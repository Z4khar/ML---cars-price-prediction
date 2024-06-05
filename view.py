# просмотр информации о датасете
from dataSet import df, train, test, a, X_train, trainG, df_train
import functionML

def information_dataset(): 
    print(df.columns) 
    print(df['Price'].value_counts()) # значения выбранного столбца
    print(df.dtypes) # типы значений 
    print(train. shape , test. shape ) # смотрим на размеры матриц (713, 15) (179, 15)
    print(df.nunique(axis=0)) # количество уникальных значений в каждом столбце 
    print(df.isnull().sum()) # проверка на наличие нулевых значений
    return()



#  просмотр информации о преобразованном датасете
information_dataset() 
functionML.IQR_outlier(train, 'Price')
print(train['Price'].describe()) #просмтор основной информации по разбросам цены на авто

functionML.display_data(df_train)
functionML.display_data(trainG) # закодированный датесет 
functionML.display_data(X_train)