# Insurance_Model_Deployment
This repository is a project I decided to create out of curiosity.All model building steps and deployments steps was done by Louis Adibe.
<br>
In the front end,you need to input values for 3 variables BMI,New Smoker and age. For the New Smoker variable,you can either input "1" or "0"
<br>
New Smoker variable:  
                  <br>
                    1-->represents smokers
                      <br>
                     0-->represents non smokers
<br>
The dataset and jupyter notebook used for building the machine learning model can also be found in this repository

### How to run the App
In order to run the python flask in your terminal type "python app.py"
<br>
# Project Flowchart
<img width="1247" alt="Project_Flowchart" src="https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/98e284df-355f-40db-a4b1-123b50e72cbd">

# Methodology
## Libraries
### Python libraries used:
Pandas(for data cleaning, data manipulation, and exploration),,Sklearn(for building regression and random forest model),Numpy(for evaluating model performance),
### Microsoft Office tools used:
Excel(Pre-data cleaning)
Powerpoint(for flowchart design)
### Backend framework used:
Flask(for web framework)
### Front-end framework used:
Bootstrap(Css&Html(for web layout styling and design)
### Azure resources used:
Azure resource group(container for holding related resources), Azure web apps(platform for building and hosting web applications)
### For continuous integration and deployments(CI/CD):
Github actions
<br>
### (1) Build a model
The dataset used in this project is a medical cost dataset, the goal here is to build a predictive model that will be able to accurately predict health insurance charges.The model was built using Multi-linear regression
<br>
#### Import all libraries that will be used
import pandas as pd
<br>
from numpy import math
<br>
import numpy as np
<br>
from sklearn.model_selection import train_test_split
<br>
from sklearn.metrics import r2_score
<br>
from sklearn.metrics import mean_squared_error
<br>
from sklearn.preprocessing import StandardScaler
<br>
from sklearn.linear_model import LinearRegression
<br>
import pickle
<br>
df=pd.read_csv('insurance.csv')
<br>
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/b2beef8c-6f1e-4a33-8763-98315ffcb5b4)
#### So drop unnecessary columns
df=df.drop(['sex', 'smoker', 'region'],axis=1)
Above I dropped columns that will not be needed for analysis
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/13f09cab-8468-4f83-b582-a69145e22b01)
#### Data Exploration
Find correlation between variables
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/ceacdb90-2d79-43f8-860d-b16187e909bd)
For better visualisation of correlation between variables,see below
<br>
from matplotlib import pyplot as plt
<br>
import seaborn as sns
<br>
plt.figure(figsize=(6,6))
<br>
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
<br>
plt.show()
<br>
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/7eb11522-643d-4832-89fa-559173c8e5ca)
From above we can observe that there is a strong correlation between the target variable Charges and New_Smoker
<br>
However, lets try to build our first model using all the independent variables and notice the output.
y=df.iloc[:,6]
y
<br>
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/539b9951-7716-4099-9fac-0105577465af)
Above is the target variable, the target variable is the Charges variable and it is the variable that we will be predicting.
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/d22715ef-895f-45e7-acd4-90e0eee34fba)
Above are the independent variables.
<br>
#### First Model
Below I have decided to split the dataset into training and test set, the test set is 30% while the training set is 70%.
<br>
I also decided to scale the dataset so that all features can have equal contributions to prediction
<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
<br>
Now lets normalize the datasets so as to enable all features have equal chance of contribution
<br>
So scale X_train and X_test
<br>
scaling=StandardScaler()
<br>
X_train=scaling.fit_transform(X_train)
<br>
X_test=scaling.fit_transform(X_test)
#### Below is a view of our training dataset
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/9ffc2497-1a0d-4164-af70-84eabfa9182d)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/87fc42e1-b882-4f9e-8120-a246173ec55a)
model = LinearRegression()  # define the linear regression model
<br>
model.fit(X_train, y_train)  # fit the data
<br>
Above I have defined the algorithm that will be used for the predictive model the algorithm is a linear regression
print('The weight vector is:', model.coef_)
<br>
print()
<br>
print('The bias is:', model.intercept_)
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/668f283e-85a5-4db2-9ad2-6f221afdea6b)



Above are two important parameters in linear regression: Weight and Bias. These two parameters determine the relationship between the independent and dependent variables (target variables).
In linear regression, the weights determine the impact of each feature on the prediction. While the Bias accounts for the offset between the predicted values and the true values.
In a good model, the bias should be very low, and as we can see the bias above is 13379.1573
However, instead of focusing only on the bias to determine a good model, it is better to evaluate the overall performance of the model. 
So to evaluate the performance of the model I will use mean squared error(MSE). The MSE is a metric for evaluating a linear regression model,it measures the average squared difference between the actual values and the predicted values
<br>
#### Compute mse
math.sqrt(mean_squared_error(y_test,y_pred))
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/0ce9d993-c8a8-493f-b83f-1f2ae161e7ac)
As we can see above the mean square is high and our goal is to minimize the MSE.
Now let's go further to evaluate the model using R-squared. The R-squared is a statistical measure that represents the proportion of variance in the target variable that is explained by the independent variables and it is always between 0 and 1.
A good R-squared should be close to 1
<br>
r2_score(y_test,y_pred)
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/b2a1c266-c9e2-4041-93e2-382e65f1bd00)
As we can see above, the R-squared is 0.76 and this is not bad because quite close to 1, but recall that this model has an unsatisfactory MSE and Bias so let us go further to build a second model to see if the model performance will improve.

#### Second Model
The reason I am trying to build a second model is that I am seeking to improve the model performance thereby improving the Bias, MSE, and R-squared.
Several reasons could be causing the poor model performance and one of them is Noise in the dataset, noise in the dataset is often due to random variability that is unrelated to the target variable, and as a result, the model is unable to distinguish between noise and signal, and if this happens, the model could decide to fit on the noise thereby leading to high MSE and high Bias.
As a strategy, I have decided to focus only on features that are closely related to the target variable.
So looking at the correlation heatmap below I can observe a strong linear correlation between the charges(target) and New_smoker

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/de697178-6513-4152-a881-03c5f1f4bd42)

y=df.iloc[:,6]
y
X=df.iloc[:,4]
X

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/91692f15-acb2-4fe4-8d8a-5e04fdecacd5)

So as seen above, I am trying to build the second model using only the New_smoker feature as the independent variable and the Charges as the dependent variable
<br>
#### I have to reshape X and y because they are both single arrays

X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaling=StandardScaler() # call scale model and pass it to variable name scaling
<br>
X_train=scaling.fit_transform(X_train) # fit the transform on x_train
<br>
X_test=scaling.fit_transform(X_test) # fit the transform on x_test
<br>
So above I have done 3 things, reshaped the data because its just 1-dimension,split the data to test and train set and scaled the train and test sets


![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/4c3c4217-6002-4435-b268-8f76ad1db866)

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/0059f6b2-e904-4cfc-9d51-d32928f25878)

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/84f6b30a-89c4-405d-9624-abcf3ec8d0d4)


![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/5eabb200-e949-4e36-b1f1-99c09aab61f3)

As can be seen above the MSE and R-squared of the second model are not satisfactory and this could be because there are insufficient features and as a result, the model is failing to capture important patterns in the data
<br>
Now I will build a third model
#### Third Model
The strategy to build the third will involve adding more relevant features, so let's use the heatmap below to find relevant features.
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/04ab70d7-0c17-48e3-a95b-68041b8ae1f2)
As can be seen from the correlation heatmap above,age, Bmi and New_Smoker have a positive correlation with charges.So I will be using those two features as dependent variables.
X=df.loc[:,['bmi','New_Smoker','age']]
X
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/2870a285-f1e2-4525-a3ff-5d016ce2bd07)


![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/b22d0d0d-71ee-49f8-8f17-180b8f7ab8fd)

As we can see above the model has improved a bit and this is partly due to the fact only correlated features are used as Independent variables.
However, I have decided to achieve a better model there improving the R-squared. This time the strategy I will apply is to utilize random forests on the same features that were used in the model 3


#### Fourth Model
In this 4th model, I will be using Random forests to build the model
and after different finetuning and iterations of parameters,I was able to get a better model

from sklearn.ensemble import RandomForestRegressor
<Br>
rf = RandomForestRegressor(n_estimators=4,max_depth=3,random_state=0)
<br>
rf.fit(X_train, y_train)

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/7277bdbc-a002-4284-9578-20096bb669fa)

After implementing random forests, I have been able to improve the model performance from 76% to 87%.

### (2) Publish a Model
Now this is the second step of the entire project
<br>
After building the model, now is the time to publish or deploy the model
<br>

### (2b) Publish model with Fast API
So in order to use fastAPi I created a virtual environment and name it fast_api_ml,below is a screenshot after the FastApi was implemented
![Fast_Api_Documentation](https://github.com/AdventureLouis/Insurance_Model_Deployment-in-Flask-and-FastAPI/assets/161846069/f8e2f912-1ab7-402c-9489-5ac0af47fafa)


I will be deploying the model to a cloud platform and for this project, I have decided to deploy the model to the AZURE cloud.

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/55218bbd-eb9a-4b8a-9f0b-33505681c754)

Above is the image of the Flask app that will be deployed to Azure.
<br>
for this deployment,I will be using GitHub actions and and Azure.
<br>
First I created a resource group in Azure
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/44f2a586-fe23-4f17-b21b-20f6d10c0139)
Above is a resource group that I created in the Azure portal. A resource group is a container for storing different resources in Azure.
Now to complete the deployment, I will create a web app resource inside the resource group

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/33d65b84-682c-435c-a086-22f8a631bdde)
<br>
Above is a web app resource created inside the resource group

![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/a89ff7cc-df0b-4449-babd-77a3e84f6e32)


Now in the deployment tab, enable continuous deployment and choose your GitHub account and repository name including the branch.
<br
Finally, review and create. After creating the web app, go to your GitHub repo and you will notice that a .github/workflows folder has automatically been created as seen below👇
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/19cfad98-1657-4829-a568-5c199ae8daaf)
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/21228a32-3ffa-44bc-8a59-c2347f73ab1d)
<br>
Next inside the .github/workflows folder, you will see a yaml file, above 👆 is the content of the yaml file that was automatically created after you deployed the web app from the Azure portal.
![image](https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/2a2785be-c8fe-4abc-9315-5c5e7b065f72)
Next while still in Github, click on the Actions tab, and you will notice the build and deploy have been successful.
<br>
Finally, click on the URL under deploy or you can get the URL from the Azure web app resource.
<br>
Below 👇 is a video demo of the deployed app

https://github.com/AdventureLouis/Insurance_Model_Deployment/assets/161846069/0a56a2cf-c24d-47ba-ab51-53179061a8ed


Run the app locally with below command:
python app.py

