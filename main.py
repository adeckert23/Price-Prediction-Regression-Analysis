import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from class_features import Features
from class_models import Linear, GLM
import scipy.stats as scs

metadata = pd.read_excel('data/Data Dictionary.xlsx')
Train = pd.read_csv('data/Train.csv')
Test = pd.read_csv('data/Test.csv')

#Target variable histograms
fig, axs=plt.subplots(1,2)
axs[0].hist(Train['SalePrice'])
axs[0].set_title('Sale Price')
axs[1].hist(np.log(Train['SalePrice']))
axs[1].set_title('Logs of Sale Price')

#Features
X=Features().features_clean(Train)
y = Train['SalePrice']

#Linear model with log transformed target variable
linear = Linear(X,np.log(y))
cv_linear = linear.cv_mse(5)
linear.resid_scatter(X,np.log(y))
linear.model_summary()

#GLM Gamma family with log-link
glm = GLM(X,y)
cv_glm = glm.cv_mse(5)
glm.resid_scatter(X,y)
glm.model_summary()

# Predictions Test data
X_test=Features().features_clean(Test)

Test['SalePrice_glm'] = glm.predict_y(X_test)
Test['SalePrice_linear'] = np.exp(linear.predict_y(X_test))

#Kernel density estimation
kde_glm = scs.gaussian_kde(Test['SalePrice_glm'])
kde_linear = scs.gaussian_kde(Test['SalePrice_linear'])

xx = np.linspace(0, 100000, 100)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(xx, kde_glm(xx), label = 'SalePrice_glm')
ax.plot(xx, kde_linear(xx), label = 'SalePrice_linear')
ax.legend()