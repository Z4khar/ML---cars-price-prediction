# Модели 
from catboost import CatBoostRegressor
from sklearn import tree
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from dataSet import* 
import numpy as np 
import functionML
import matplotlib.pyplot as plt

# Читаем файлы с данными
VERSION = 8
#test_upd = pd.read_csv(DIR_TEST_UPD+'auto_test.csv') #датасет для обучения модели

sample_submission = test

# Стандартная нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #обучаем на тренеровочной выборке
X_test_scaled = scaler.transform(X_test) # передаем коэффициенты в тестовую

def mape(y_true, y_pred):
    '''
    Метрика
    '''
    return np.mean(np.abs((y_pred-y_true)/y_true))

mape_scorer = make_scorer(
    mape, 
    greater_is_better=False
)
"""
VAL_SIZE   = 0.20   # 20% 
RANDOM_SEED = 42
"""

#function.display_data(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)

# 1. CatBoost
def Cat_Boost_Model(X_train, X_test, target_train, target_test):
    
    model = CatBoostRegressor(iterations = 50,
                              random_seed = RANDOM_SEED,
                              eval_metric='MAPE',
                              custom_metric=['R2', 'MAE'],
                              silent=True,
                             )
    model.fit(X_train, target_train,
             cat_features=categorical_features_indices,
             eval_set=(X_test, target_test),
             verbose_eval=0,
             use_best_model=True
             )
    return model

def _Model(name_model, X_train, X_test, LogTarget=False, **kwargs):
    '''
    Обоолочка для моделей
    '''   
    
    if LogTarget:
        target_train = np.log(y_train)
        target_test = np.log(y_test)
    else:
        target_train = y_train
        target_test = y_test
    
    model = eval(name_model+'(X_train, X_test, target_train, target_test, **kwargs)')
    
    #model.save_model('catboost_single_model_baseline.model')
    
    # оцениваем точность
    if LogTarget:
        predict = np.exp(model.predict(X_test)) 
    else:    
        predict = model.predict(X_test)
    
    return model, round((mape(y_test, predict))*20, 3) 

model, mape_ =_Model('Cat_Boost_Model', X_train, X_test)
results['Cat_Boost_Model'] = [mape_, model.get_params()]
best_model = 'Cat_Boost_Model'
print(f"Точность модели по метрике MAPE (Сatboost): {mape_:0.2f}%")

"""
#Log Target - позволит уменьшить влияние выбросов на 
#обучение модели (используем для этого np.log и np.exp)
"""
model, mape_ = _Model('Cat_Boost_Model',X_train, X_test, LogTarget=True)
results['Cat_Boost_Model_target_log'] = [mape_, model.get_params()]
predict_submission = model.predict(X_sub)
print(f"Точность модели по метрике MAPE (Сatboost): {mape_:0.2f}%") 

# 2. Наивная модель 
# Наивная модель - с ней сравниваются другие модели 
tmp_train = X_train.copy()
tmp_train['Price'] = y_train

# Находим median по экземплярам engineDisplacement в трейне и размечаем тест
predict = X_test['Engine Capacity(L)'].map(tmp_train.groupby('Engine Capacity(L)')['Price'].median())

#оцениваем точность
mape_ = round((mape(y_test, predict.values))*20, 3) # 100, 3
results['Наивная модель'] = [mape_, '']
print(f"Точность наивной модели по метрике MAPE: {mape_:0.2f}%")

# 3. Линейная регрессия 
def Linear_Regression_Model(X_train, X_test,target_train, target_test):
    
    model = LinearRegression()
    model.fit(X_train, target_train)
    return model


# линейная регрессия
model, mape_ = _Model('Linear_Regression_Model', X_train, X_test, LogTarget=False)
results['Linear_Regression_Model'] = [mape_, model.get_params()]
y_pred = model.predict(X_test)
print(f"ЛР Точность модели по метрике MAPE: {mape_:0.2f}%")

if (mape_ < results[best_model][0]): 
    best_model = 'Linear_Regression_Model'   

#y_pred = y_pred*10
print(y_pred)

pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
print(pred_df.head(5))
 

# линейная регрессия с логарифмическим таргетом
model, mape_ = _Model('Linear_Regression_Model', X_train, X_test, LogTarget=True)
results['Linear_Regression_Model_target_log'] = [mape_, model.get_params()]

print(f"ЛР Точность модели по метрике MAPE: {mape_:0.2f}%")
if (mape_ < results[best_model][0]): 
    best_model = 'Linear_Regression_Model_target_log'

scales = pd.Series(data=X_train.std(axis=0), index=X.columns)
functionML.show_weights(X.columns, model.coef_, scales)
plt.title('Линейная регрессия')
plt.show()


# отбор признаков на основе модели 
select = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42),threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("форма обуч набора X: {}".format(X_train.shape))
print("форма обуч набора X c l1: {}".format(X_train_l1.shape))

mask = select.get_support()
print(mask)
# визуализируем булевы значения: черный – True, белый – False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Индекс примера")

mask2 = select.get_support()
# визуализируем булевы значения -- черный – True, белый – False
plt.matshow(mask2.reshape(1, -1), cmap='gray_r')
plt.xlabel("Индекс примера")
plt.show()

# линейная регрессия + mask + log
model, mape_ = _Model('Linear_Regression_Model'\
                      , X_train[X_train.columns[mask2].tolist()]
                        , X_test[X_test.columns[mask2].tolist()]
                        , LogTarget=True
)
results['Linear_Regression_Model_mask_log'] = [mape_, model.get_params()]

print(f"Точность модели по метрике MAPE: {mape_:0.2f}%")
if (mape_ < results[best_model][0]): 
    best_model = 'Linear_Regression_Model_mask_log'  

alphas = np.logspace(-1, 6, 100)
searcher = GridSearchCV(Ridge(), [{"alpha": alphas}], scoring=mape_scorer, cv=10)
searcher.fit(X_train_scaled, np.log(y_train))

best_alpha = searcher.best_params_["alpha"]
print("Best alpha = %.4f" % best_alpha)
plt.figure(figsize=(5, 5))
plt.plot(alphas, -searcher.cv_results_["mean_test_score"])
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("CV score")
plt.show()

# Лассо

def Lasso_Pipeline(X_train, X_test,target_train, target_test, **kwargs):
    
    alpha = kwargs['alpha']
    
    lasso_pipeline = Pipeline(steps=[
        ('scaling', StandardScaler()),
        ('regression', Lasso(alpha=alpha, max_iter=4000))
    ])

    model = lasso_pipeline.fit(X_train, target_train)
    
    return model

model, mape_ = _Model('Lasso_Pipeline', X_train, X_test, LogTarget=False, alpha=best_alpha)
results['Lasso_Pipelinel_scaled_bestalpha'] = [mape_, model.get_params()]

print(f"Точность модели по метрике MAPE: {mape_:0.2f}%")

#if (mape_ < results[best_model][0]): 
#    best_model = 'Lasso_Pipeline_scaled_bestalpha' 



# случайный лес 

def RandomForestRegressor_Model(X_train, X_test,target_train, target_test, **kwargs):
       
    model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **kwargs)
    model.fit(X_train, target_train)
    return model

model, mape_ = _Model('RandomForestRegressor_Model'
                      , X_train
                      , X_test
                      , LogTarget=False
                      , n_estimators=400
                      , min_samples_split=2
                      , min_samples_leaf=1
                      , max_features=X_train.shape[1]//3
                      , max_depth=None
                      , bootstrap=False)

results['RandomForestRegressor_Model_400'] = [mape_, model.get_params()]

print(f"Точность модели по метрике MAPE: {mape_:0.2f}%")
if (mape_ < results[best_model][0]): 
    best_model = 'RandomForestRegressor_Model_400' 



# отображение дерева
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, df_train)
plt.figure(figsize=(20,20))
_ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)
plt.show()


# XGB regression
"""
xgbr = XGBRegressor()

def XGBRegressor_Model(X_train, X_test,target_train, target_test, **kwargs):
       
    model = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, **kwargs)
    model.fit(X_train, target_train)
    return model

model, mape_ = _Model('XGBRegressor_Model'
                      , X_train
                      , X_test
                      , LogTarget=False
                      )

results['XGBRegressor_Model_default'] = [mape_, model.get_params()]

print(f"Точность модели по метрике MAPE: {mape_:0.2f}%")

if (mape_ < results[best_model][0]): 
    best_model = 'XGBRegressor_Model_default' 

    

model = XGBRegressor()
n_estimators = range(100, 1000, 100)
param_grid = dict(n_estimators=n_estimators)
kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
grid_search = GridSearchCV(model, param_grid, scoring=mape_scorer, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# plot
plt.figure(figsize=(6, 6))
plt.errorbar(n_estimators, means, yerr=stds)
plt.title("XGBoost n_estimators vs MAPE")
plt.xlabel('n_estimators')
plt.ylabel('MAPE')
plt.show()


model, mape_ = _Model('XGBRegressor_Model'
                      , X_train
                      , X_test
                      , LogTarget=False
                      , n_estimators=500
                      )

results['XGBRegressor_Model_500'] = [mape_, model.get_params()]

print(f"Точность модели по метрике MAPE: {mape_:0.2f}%")
if (mape_ < results[best_model][0]): 
    best_model = 'XGBRegressor_Model_500' 


model = XGBRegressor()
n_estimators = range(200, 1300, 200)
max_depth = [4, 6, 8, 10]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
grid_search = GridSearchCV(model, param_grid, scoring=mape_scorer, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# plot
plt.figure(figsize=(6, 6))
scores = np.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.plot(n_estimators, scores[i], label='depth: ' + str(value))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('MAPE')
plt.show()
"""
"""
# Заключительный вывод
y_pred = model.predict(X_test)
print(y_pred)

pred_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
pred_df
"""





