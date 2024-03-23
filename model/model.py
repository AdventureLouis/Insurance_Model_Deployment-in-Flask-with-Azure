import pandas as pd
from sklearn import linear_model
import pickle

df=pd.read_csv('venv\model\insurance.csv' )
y=df.loc[:,'charges']
X=df.loc[:,['bmi','New_Smoker']]

lm = linear_model.LinearRegression()

lm.fit(X,y)

#lm.predict([[15, 61]])


pickle.dump(lm, open('model.pkl','wb')) 


print(lm.predict([[15, 61]]))  # format of input
print(f'score: {lm.score(X, y)}')
