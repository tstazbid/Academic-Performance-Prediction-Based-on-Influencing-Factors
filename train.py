import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('dataset/student-mat.csv')
#df2 = pd.read_csv('dataset/student-por.csv')
#common_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
#df = pd.merge(df1, df2, on=common_columns, how='inner')

print(df.head())
df['final_grade'] = df['G3'].apply(lambda x: 'high' if 10 <= x <= 20 else ('low' if 0 <= x <= 10 else 'na'))
cleanup_nums = {
    "school":{"GP": 0, "MS": 1},
    "sex":{"F": 0, "M": 1},
    "address":{"U": 0, "R": 1},
    "famsize":{"LE3": 0, "GT3": 1},
    "Pstatus":{"T": 0, "A": 1},
    "Mjob":{"teacher": 0, "health": 1,"services": 2,"at_home": 3,"other": 4},
    "Fjob":{"teacher": 0, "health": 1,"services": 2,"at_home": 3,"other": 4},
    "reason":{"home": 0, "reputation": 1,"course":2,"other":3},
    "guardian":{"mother": 0, "father": 1,"other":2},
    "schoolsup":{"yes": 0, "no": 1},
    "famsup":{"yes": 0, "no": 1},
    "paid":{"yes": 0, "no": 1},
    "activities":{"yes": 0, "no": 1},
    "nursery":{"yes": 0, "no": 1},
    "higher":{"yes": 0, "no": 1},
    "internet":{"yes": 0, "no": 1},
    "romantic": {"yes": 0, "no": 1},
    "final_grade":{"high":0, "low":1}}

df.replace(cleanup_nums, inplace=True)
df = df.drop('G3', axis=1)
print(df.head())

X = df.drop('final_grade',axis=1)
y = df['final_grade']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn import metrics

def predict(ml_model):
    model = ml_model.fit(X_train,y_train)
    print(model)
    
from sklearn.ensemble import RandomForestRegressor
randomForest = RandomForestRegressor();
predict(randomForest)

pickle.dump(randomForest, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))