import pandas as pd 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv('venv\model\insurance_1.csv' )
X=df[['bmi','New_Smoker','age']]
y=df['charges']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# So scale X_train and X_test
scaling=StandardScaler()
X_train=scaling.fit_transform(X_train)
X_test=scaling.fit_transform(X_test)

rf = RandomForestRegressor(n_estimators=4,max_depth=3,random_state=0)
rf.fit(X_train, y_train)

#lm.predict([[15, 61]])


pickle.dump(rf, open('model.pkl','wb')) 


# print(rf.predict(X))  # format of input
# print(f'score: {rf.score(X, y)}')
