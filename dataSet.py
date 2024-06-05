# Работа с датасетом
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from dictionary import*
import functionML
#from Setup import*


VAL_SIZE   = 0.20   # 20% 
RANDOM_SEED = 42

results = {} # models with results
predict_submission, best_model = '', '' # submission, best_model

# словари
cat_features = ['Model', 'bodyType', 'Color', 'FuelType',
               'numberOfDoors', 'Transmission',
                'Cylinder', 'SellerRating', 'Drivetrain'] 
num_features = [ "HorsePower", 'Mileage', 'Engine Capacity(L)', 'Price', 'FuelRate', 'Year']

# импорт исходного датасета 
filepath = r"D:/Programming/MLcars/ML.csv"; # датасет для обучения
#filepathT = r"D:/Programming/MLcars/MLtest2.csv"; # тестовый датасет 
df = pd.read_csv(filepath)


# удаляем ненужные столбцы
df.drop(['SellerType', 'SellerName',
         'SellerReviews', 'StreetName', 'State', 
         'Zipcode', 'VIN', 'Stock#', 
          'ComfortRating', 'InteriorDesignRating', 'PerformanceRating', 
          'ValueForMoneyRating', 'ExteriorStylingRating', 'ReliabilityRating',
          'MinMPG', 'Engine', 'Make', 'ConsumerReviews', 'InteriorColor', 'DealType'], axis= 1 , inplace= True ) 

# удаляем пропущенные значения 
df.dropna(axis=0, inplace=True) # строки 
df.dropna(axis=1, inplace=True) # столбцы

# преобразуем цену, пробег и расход топлива  
df.Mileage = df.Mileage*1.609 # км
#df.Price = df.Price*90 # руб
df.MaxMPG = df.MaxMPG*0.43 # л/км
df.rename(columns = {'MaxMPG': "FuelRate"}, inplace = True)

# преобразуем цвета автомобиля 
"""Черный"""
df['Color'].replace(['Jet Black','Black Sapphire','Carbon Black Metallic', 
                     'Black Sapphire Metallic','Black Sapphire Metallic', 'Azurite Black Metallic', 
                       'Ruby Black Metallic','Dark Graphite Metallic'], 'Black', inplace=True)
"""Белый"""
df['Color'].replace(['Alpine White','Mineral White Metallic','Mineral White Metallic','White Metallic'], 'White', inplace=True)
"""Синий"""
df['Color'].replace(['Phytonic Blue Metallic','Bluestone Metallic','Imperial Blue Metallic',
                      'San Marino Blue Metallic','San Marino Blue Metallic','Portimao Blue Metallic', 
                        'Mediterranean Blue Metallic','Estoril Blue Metallic','Estoril Blue Metallic', 
                           'Blue Ridge','Bay Blue','Blue Ridge Mountain Metallic','Blue Metallic',
                           'Tanzanite Blue Ii Metallic','Tanzanite Blue II Metallic','Tanzanite Blue II Metallic'], 'Blue', inplace=True)
"""Серый"""
df['Color'].replace(['Arctic Gray Metallic','Gray Metallic','Mineral Gray Metallic','Space Gray Metallic','Magellan Gray Metallic',
                     'Mineral Gray','Space Gray Metallic','Metallic','Dark Graphite',
                     'Storm Bay Metallic','GRAY','Sunstone Metallic'], 'Gray', inplace=True)
"""Серебрянный"""
df['Color'].replace(['Glacier Silver Metallic','Mineral Silver Metallic','Titanium Silver Metallic',
                     'Platinum Silver Metallic','Glacier Silver','Silver Metallic'], 'Silver', inplace=True)
"""Ораньжевый"""
df['Color'].replace(['Sunset Orange Metallic','Sakhir Orange II Metallic','–'], 'Orange', inplace=True)
"""Оливковый"""
df['Color'].replace(['Dark Olive','Dark Olive Metallic'], 'Olive', inplace=True)
"""Коричневый"""
df['Color'].replace(['Brown Metallic','Sparkling Brown Metallic','Bronze Metallic',
                     'Chestnut Bronze','Vermont Bronze Metallic'], 'Brown', inplace=True)
"""Красный"""
df['Color'].replace(['Flamenco Red Metallic','Melbourne Red Metallic','Ametrin Metallic','Burgundy'], 'Red', inplace=True)


# разделяем полученный датасет на две части: тестовую и обучающую выборки с помощью sclearn
train, test = train_test_split(df, test_size= 0.2 , random_state= 0 )

# обрезаем тренировочный датасет до 20 моделей
train2 = train[train['Model'].isin(train['Model'].value_counts()[:20].index.tolist())]

"""
trainPlt = train

#Усечение датасета
trainPlt.drop(index=list(train[train['Price']<200000].index)+
           list(train[train['Price']>7000000].index),axis=0, inplace=True)
"""
train['sample'] = 1 # помечаем где у нас обучающая выборка
test['sample'] = 0 # помечаем где у нас тестовая выборка

data = train
y = data[data['sample']==1]['Price']
X = data.query('sample == 1').drop(['sample'], axis=1)
X_sub = data.query('sample == 0').drop(['sample'], axis=1)

categorical_features_indices = list(set(X.columns)-set(['Price','HorsePower','Mileage','Year'])-\
    set([x+'_c' for x in cat_quality_values.keys()]))

X = X[categorical_features_indices]
categorical_features_indices = np.where(X.dtypes != np.float64)[0]
len(categorical_features_indices)

# Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)

# Кодирование данных 
label_encoder = LabelEncoder()

data['Color']= label_encoder.fit_transform(data['Color'])
data['Drivetrain']= label_encoder.fit_transform(data['Drivetrain'])
data['FuelType']= label_encoder.fit_transform(data['FuelType'])
data['Model']= label_encoder.fit_transform(data['Model'])
data['bodyType']= label_encoder.fit_transform(data['bodyType'])
data['Transmission']= label_encoder.fit_transform(data['Transmission'])

X_train['Color']= label_encoder.fit_transform(X_train['Color'])
X_train['Drivetrain']= label_encoder.fit_transform(X_train['Drivetrain'])
X_train['FuelType']= label_encoder.fit_transform(X_train['FuelType'])
X_train['Model']= label_encoder.fit_transform(X_train['Model'])
X_train['bodyType']= label_encoder.fit_transform(X_train['bodyType'])
X_train['Transmission']= label_encoder.fit_transform(X_train['Transmission'])

X_test['Color']= label_encoder.fit_transform(X_test['Color'])
X_test['Drivetrain']= label_encoder.fit_transform(X_test['Drivetrain'])
X_test['FuelType']= label_encoder.fit_transform(X_test['FuelType'])
X_test['Model']= label_encoder.fit_transform(X_test['Model'])
X_test['bodyType']= label_encoder.fit_transform(X_test['bodyType'])
X_test['Transmission']= label_encoder.fit_transform(X_test['Transmission'])


# окончательная подготовка датасетов после кодирования 

df_train = data[data['sample']==1]
df_test = data[data['sample']==0]
df_train = df_train.iloc[0:583] # обрезка

#functionML.display_data(X_test)
#functionML.display_data(df_train)

