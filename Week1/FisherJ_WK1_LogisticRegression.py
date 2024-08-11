#import libraries
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#fetch data
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)

#separate features and target 
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets

#flatten y into a 1D array
y = y.to_numpy().ravel()

#check for NaN values
print('Number of NaN values in X: ', X.isnull().sum())

#replace NaN values of 'Bare_nuclei' feature column with the mode of that column

print('Number of NaN values in X: ', np.isnan(X).sum())
X['Bare_nuclei'] = X['Bare_nuclei'].fillna(X['Bare_nuclei'].mode()[0])

#check for NaN values
print('Number of NaN values in X: ', np.isnan(X).sum())


### CREATE Logistic Regression Model ###

#split data into training and testing sets (testing = 25% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

#initialize the model
model = LogisticRegression()

#fit the model
model.fit(X_train, y_train)

#predict the test set
y_pred = model.predict(X_test)

#calculate accuracy
accuracy = accuracy(y_test, y_pred)
print('Accuracy: ', accuracy)

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ')
print(conf_matrix)

#display confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()