# План проекта
  
## Название проекта:
Сохранение клиентов с использованием анализа данных и построения моделей.
    
    
## Описание проекта:
 Оператор связи  хочет научиться прогнозировать отток клиентов. Если выяснится, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и договорах.
    
## Цели и задачи проекта: 
Найти и обучить модель для предсказания возможного ухода клиентов. Создание качественно метрики качества и предоставление интерпретируемой метрики для доклада(метрика должна быть легко объяснена). ROC-AUC > 0.85 на тестовой выборке.
    
    
## Детальный план разработки модели:
   
    1. Провести исследовательский анализ данных
    
        1.1 Первичный взгляд на данные с использованием графиков, просмотров типов данных
    
    2. Предобработка данных и создание модели
         
        2.1 Соединение таблиц
    
        2.2 Расширение признакового пространства
    
        2.3 Вычисление корреляции и удаление сильнокоррелирующих и некоррелирующих признаков
        
        2.4 Разбиение выборок на обучающую и тестовую в соотношении 3:1
    
        3. Создание модели 
        
        3.1 Испытание моделей(CatBoost, LinearRegressor, RandomForest)
        
        3.2 Pipelint(масштабирование, кодирование признаков)
        
        3.3 Проверка лучшей модели на тестовой выборке
    
        3.4 Построение графика ROC
    
        3.5 Анализ матрицы ошибок 
    
        3.6 Анализ важности признаков 
    
## Используемые библиотеки:
```python
import pandas as pd

import numpy as np 

import phik

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

from sklearn.compose import make_column_transformer

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    train_test_split)

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler)

