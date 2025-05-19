# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree regressor
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Syed Mohamed Raihan M
RegisterNumber:  212224240167
*/

import pandas as pd
df=pd.read_csv("/content/Salary.csv")
df.head()
df.info()
df.isnull().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()


x=df[['Position','Level']]
x.head()
y=df['Salary']
y.head()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
r2


dt.predict([[5,6]])

```

## Output:

![Screenshot 2025-05-19 100328](https://github.com/user-attachments/assets/4fd6aff6-e3e2-4d8b-8b70-8037f837f325)
![Screenshot 2025-05-19 100336](https://github.com/user-attachments/assets/372a4785-25eb-4049-b022-4590e4eb1983)
![Screenshot 2025-05-19 100352](https://github.com/user-attachments/assets/679f31a8-a908-4ae5-98ae-ded767e3aa0d)
![Screenshot 2025-05-19 100411](https://github.com/user-attachments/assets/d3f78b84-d541-41a9-b992-25b8d3ccd4aa)
![Screenshot 2025-05-19 100416](https://github.com/user-attachments/assets/0637f631-9582-4103-8a3c-4b37ff696e09)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
