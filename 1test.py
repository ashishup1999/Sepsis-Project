# Testing on the saved model
import pandas as pd
import numpy as np

    
# Importing dataset
dataset = pd.read_csv('p119997.psv', sep='|')
X = dataset.iloc[:,0:40].values
y = dataset.iloc[:,40:41].values

# Imputing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
X = imputer.fit_transform(X)
