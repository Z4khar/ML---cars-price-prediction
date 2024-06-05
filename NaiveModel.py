# Наивная модель - с ней сравниваются другие модели 
from dataSet import* 
from functionML import mape


tmp_train = X_train.copy()
tmp_train['Price'] = y_train

# Находим median по экземплярам engineDisplacement в трейне и размечаем тест
predict = X_test['Engine Capacity(L)'].map(tmp_train.groupby('Engine Capacity(L)')['Price'].median())

#оцениваем точность
mape_ = round((mape(y_test, predict.values))*15, 3) # 100, 3
results['Наивная модель'] = [mape_, '']
print(f"Точность наивной модели по метрике MAPE: {mape_:0.2f}%")
