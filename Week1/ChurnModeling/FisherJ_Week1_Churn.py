#load libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#load data
df = pd.read_csv('C:/Users/unkno/Desktop/MS Data Science/Class 9 - ANA680/Week 1/assignment2/Churn_Modelling.csv')

#check data
print(df.head())
print(df.info())
print(df.shape)

## PREPARE DATA
#encode categorical data (geography and gender)
geoEncoder = LabelEncoder()
df['Geography'] = geoEncoder.fit_transform(df['Geography'])

genderEncoder = LabelEncoder()
df['Gender'] = genderEncoder.fit_transform(df['Gender'])

#print first 5 rows of geography and gender
print(df[['Geography','Gender']].head())

#assess correlation between variables for feature selection
corr_df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
corr_df.corr()
#                   Exited
# CreditScore     -0.027094   <-- low correlation, below 0.03
# Geography       -0.035943
# Gender          -0.106512
# Age              0.285323
# Tenure          -0.014001   <-- low correlation, below 0.03
# Balance          0.118533
# NumOfProducts   -0.047820
# HasCrCard       -0.007138   <-- low correlation, below 0.03
# IsActiveMember  -0.156128
# EstimatedSalary  0.012097   <-- low correlation, below 0.03

#split data into feature and target variables, dropping irrelevant columns and features with low correlation (< 0.03)
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Tenure', 'HasCrCard', 'EstimatedSalary', 'Exited'], axis=1)
y = df['Exited']

#split data into training and test partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

#scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## BUILD AND TRAIN MODEL
#initialize the artificial neural network (ANN) model
ann_model = Sequential()

#add input and first hidden layer
ann_model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))

#add second hidden layer
ann_model.add(Dense(units=8, activation='relu'))

#add output layer
ann_model.add(Dense(units=1, activation='sigmoid'))

#compile ANN model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train ANN model
ann_model.fit(X_train, y_train, batch_size=16, epochs=100)

#predict test data
y_pred = (ann_model.predict(X_test) > 0.5).astype("int32")

## EVALUATE MODEL
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_matrix)

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")