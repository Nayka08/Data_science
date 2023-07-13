# План проекта
  
## Название проекта:
Исследование объявлений о продаже квартир
    
    
## Описание проекта:
Сервис по продаже автомобилей с пробегом разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В нашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Нужно построить модель для определения стоимости. 


## Цели и задачи проекта: 
    - качество предсказания, метрика RMSE < 1800;
    - скорость предсказания;
    
    
## Детальный план разработки модели:
    1  Анализ данных
        1.1  Дата скачивания анкеты
        1.2  Год регистрации
        1.3  Исследуем целевой признак
        1.4  Зависимость цены от бренда
        1.5  Промежуточный вывод
    2  Предобработка данных
        2.1  Заполним пропуски и проверим наличие дубликатов
        2.2  Заполним нулевые значения
        2.3  Удалим неинформативные признаки
        2.4  Разделение данных
    3  Обучение моделей
        3.1  Масштабирование и стандартизация
        3.2  LightGBM
        3.3  Catboost
        3.4  Stochastic Gradient Descent
    4  Анализ моделей
    5  Тестирование лучшей модели
    6  Важность признаков
    7  Вывод    

## Используемые библиотеки:
```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    train_test_split)
