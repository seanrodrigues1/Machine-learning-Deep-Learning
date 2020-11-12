# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# PREDICTING HEART DISEASE BASED ON SEVERAL FEATURES

import pandas as pd

heart=pd.read_csv(r'C:\Users\ADMIN\Desktop\heart.csv')
print(heart)

#data exploration

# checking data types of the features in the heart data
print(heart.dtypes)
print(heart.shape)
print(heart.info())

print(heart.describe())

#check for missing values
print(heart.isnull().sum())

#data exploration 

#outcome variable distribution

import matplotlib.pyplot as plt
import seaborn as sns


target_counts=heart['target'].value_counts()
print(target_counts)

sns.countplot(data=heart,x='target')
plt.show()

#sex feature distribution
target_counts=heart['sex'].value_counts()
print(target_counts)
sns.countplot(data=heart,x='sex')
plt.show()

#age distribution

plt.hist(data=heart,x='age')
plt.xticks([20,30,40,50,60,70,80])
plt.show()

#Fasting blood sugar >120 gm (fbs) (1=TRUE,0=FALSE)

sns.countplot(data=heart,x='fbs')
plt.show()

# Heart disease frequency according to fbs

pd.crosstab(heart.fbs,heart.target).plot(kind="bar",color=['r','g' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

#chest pain type (cp)
sns.countplot(data=heart,x='cp')
plt.show()

#heart disease frequency based on chest pain type

pd.crosstab(heart.cp,heart.target).plot(kind="bar",color=['g','r' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


#normalizng the numeric features for better results

from sklearn import preprocessing




column_names_to_normalize = ['age','trestbps','chol','thalach','oldpeak']
x = heart[column_names_to_normalize].values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = heart.index)
print(df_temp)

#replacing old values with scaled values

df1=heart.drop(['age','trestbps','chol','thalach','oldpeak'],axis=1)

print(df1.info())

df1['age']=df_temp['age']
df1['trestbps']=df_temp['trestbps']
df1['chol']=df_temp['chol']
df1['thalach']=df_temp['thalach']
df1['oldpeak']=df_temp['oldpeak']
print(df1.info())


#creating dummy variables for the categorical features

a = pd.get_dummies(df1['cp'], prefix = "cp")
b = pd.get_dummies(df1['thal'], prefix = "thal")
c = pd.get_dummies(df1['slope'], prefix = "slope")
frames = [df1, a, b, c]
df_final = pd.concat(frames, axis = 1)
print(df_final.info())

heart1=df_final.drop(['cp','thal','slope'],axis=1)
print(heart1.info())


#splitting the dependent and idenpendent features and storing them in separate tables

X=heart1.drop(['target'],axis=1)
Y=heart1['target']

print(X)
print(Y)


#training test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

print(x_train)
print(x_test)

print(y_train)
print(y_test)

#running machine learning models

#logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

logistic_regression= LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred=logistic_regression.predict(x_test)


confusion_matrix = pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)
#overall accuracy

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

#sensitivity and specificity

TPR=25/(25+8)
TNR=38/(38+5)
print(TPR)
print(TNR)

#decision tree

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_hat = clf.predict(x_test)

confusion_matrix = pd.crosstab(y_hat, y_test, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)

#overall accuracy
print('Accuracy: ',metrics.accuracy_score(y_test, y_hat))

print(classification_report(y_test, y_hat))

TPR=27/(27+6)
TNR=31/(31+12)
print(TPR)
print(TNR)

#random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_hat1 = rf.predict(x_test)

confusion_matrix = pd.crosstab(y_hat1, y_test, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test, y_hat1))

print(classification_report(y_test,y_hat1))

TPR=26/(26+7)
TNR=38/(38+5)

print(TPR)
print(TNR)


#knn

from sklearn.neighbors import KNeighborsClassifier 
  
knn = KNeighborsClassifier(n_neighbors =33) 
  
knn.fit(x_train, y_train) 
pred1 = knn.predict(x_test) 

confusion_matrix = pd.crosstab(pred1, y_test, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test, pred1))

print(classification_report(y_test,pred1))

TPR=27/(27+6)
TNR=37/(37+6)
print(TPR)
print(TNR)

#naive baise

from sklearn.naive_bayes import GaussianNB
 

naive_bayes = GaussianNB()
 
#Fitting the data to the classifier
naive_bayes.fit(x_train , y_train)
 
#Predict on test data
y_predicted = naive_bayes.predict(x_test)

confusion_matrix = pd.crosstab(y_predicted, y_test, rownames=['predicted'], colnames=['actual'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test, y_predicted))

print(classification_report(y_test,y_predicted))

#svm

from sklearn.svm import SVC 
 
svm = SVC(kernel='linear') 
svm.fit(x_train, y_train) 
y_hat2=svm.predict(x_test)

confusion_matrix = pd.crosstab(y_hat2, y_test, rownames=['predicted'], colnames=['actual'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test, y_hat2))

print(classification_report(y_test,y_hat2))

#svm using rgb kernel

svm = SVC(kernel='rbf', random_state=1, gamma=1, C=1) 
svm.fit(x_train, y_train) 
y_hat3=svm.predict(x_test)

confusion_matrix = pd.crosstab(y_hat3, y_test, rownames=['predicted'], colnames=['actual'])
print(confusion_matrix)

print('Accuracy: ',metrics.accuracy_score(y_test, y_hat3))

print(classification_report(y_test,y_hat3))


