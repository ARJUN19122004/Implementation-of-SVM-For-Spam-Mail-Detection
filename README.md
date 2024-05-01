# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1:start
step 2:Import the required libraries.
step 3:Read the data frame using pandas.
step 4:Get the information regarding the null values present in the dataframe.
step 5:Split the data into training and testing sets.
step 6:Convert the text data into a numerical representation using CountVectorizer.
step 7:Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
step 8:Finally, evaluate the accuracy of the model.
step 9:stop
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.ARJUN
RegisterNumber: 212222040012

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
*/
```

## Output:
## RESULT OUTPUT
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/74cbedd2-c511-478b-bde3-b12725f7c980)
## data.head()
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/95225c83-f4e5-49b5-96e4-15f479df9464)
## data.info()
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/d8688f5f-06b8-42f8-b819-b661e1f90640)
## data.isnull.sum()
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/fd12b725-881a-40a6-abd6-a83d7a433d6d)
## Y _prediction value
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/6a1e50ab-1865-44d1-bb5e-2c6c936c31e9)
## Accuracy value
![image](https://github.com/ARJUN19122004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119429483/38547784-b460-4d6a-af91-956f226dce4b)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
