# План проекта
  
## Название проекта:
Токсичные комментарии
    
    
## Описание проекта:
Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.

    
## Цели и задачи проекта: 
    Постройте модель со значением метрики качества *F1* не меньше 0.75. 
    
    
## Детальный план разработки модели:
   
    1  Подготовка

        1.1  Первичный анализ

        1.2  Лемматизация

        1.3  Разделение на выборки

    2  Обучение

        2.1  Логистическая регрессия

        2.2  LightGBM

    3  Проверка модели

    4  Выводы


    
## Используемые библиотеки:
```python
import pandas as pd

import numpy as np

import nltk

import re

import lightgbm as lgb

from tqdm import tqdm

from nltk.corpus import wordnet

from nltk.corpus import stopwords as nltk_stopwords

from nltk.stem import WordNetLemmatizer 

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer 
