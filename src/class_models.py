import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class Linear():
    def __init__(self,X,y):
        self.X = X
        self.y = y
    
    def linear_model(self):
        model = LinearRegression()
        model.fit(self.X,self.y)
        return model
    
    def predict_y(self,X):
        model = self.linear_model()
        y_predicted = model.predict(X)
        return y_predicted
    
    def cv_mse(self,k):
        cv_mses = (cross_val_score(self.linear_model(),self.X,self.y,cv=k,
                                   scoring='neg_mean_squared_error'))*-1
        return np.mean(cv_mses)
        
    def resid_scatter(self,X,y):
        y_predicted = self.predict_y(X)
        plt.scatter(y_predicted,y-y_predicted)
    
    def model_summary(self):
        X = sm.add_constant(self.X)
        model = sm.OLS(self.y,X)
        model=model.fit()
        return model.summary()
    


class GLM():
    def __init__(self,X,y):
        self.X = X
        self.y = y
    
    def glm_model(self):
        self.X = sm.add_constant(self.X)
        model = sm.GLM(self.y,self.X,family=
                       sm.families.Gamma(link=sm.families.links.log))
        model=model.fit()
        return model
    
    def predict_y(self,X):
        model = self.glm_model()
        X = sm.add_constant(X)
        y_predicted = model.predict(X)
        return y_predicted
    
    def cv_mse(self,k):
        X = sm.add_constant(self.X)
        X=X.values
        kf = KFold(n_splits=k)
        mses=[]
        for train_index, test_index in kf.split(X):
            model = sm.GLM(self.y[train_index],X[train_index],family=
                           sm.families.Gamma(link=sm.families.links.log))
            model=model.fit()
            test_predicted = model.predict(X[test_index])
            mses.append(mean_squared_error(np.log(self.y[test_index]),np.log(test_predicted)))    
            return np.mean(mses)
        
    def resid_scatter(self,X,y):
        y_predicted = self.predict_y(X)
        plt.scatter(np.log(y_predicted),np.log(y)-np.log(y_predicted))
    
    def model_summary(self):
        model = self.glm_model()
        return model.summary()