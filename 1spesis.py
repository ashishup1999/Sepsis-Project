import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset importing

x=(np.array([0]*40)).reshape(1,-1)
y= np.array([0])

for i in range(100001,120001):
    dataset = pd.read_csv('p'+str(i)+'.psv',sep = '|')
    x = np.vstack((x,dataset.iloc[:,:-1].values))
    y = np.vstack((y,dataset.iloc[:,40:41].values))
    
X = x[1:,:]
y = y[1:,:]


# Imputing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
X = imputer.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Saving model 
import pickle
with open('model.pkl', 'wb') as file:  
    pickle.dump(classifier, file)




