import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('Iris.csv')

x = df.iloc[:,1:-1]
le = LabelEncoder()

y = le.fit_transform(df.iloc[:,-1])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=10)

LR = LinearRegression()

LR.fit(x_train,y_train)

with open("LR.pkl", 'wb') as f:
    pickle.dump(LR,f)