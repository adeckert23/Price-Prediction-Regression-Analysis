import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


df = pd.read_excel('data/Data Dictionary.xlsx')
Train = pd.read_csv('data/Train.csv')
Test = pd.read_csv('data/Test.csv')
median = pd.read_csv('data/median_benchmark.csv')

def features_func(df):

    #Columns to include
    interested_features = df[['SalesID',
       'YearMade', 'MachineHoursCurrentMeter', 'UsageBand', 'saledate',
       'ProductSize','fiProductClassDesc','state',
       'ProductGroup', 'Drive_System', 'Enclosure',
       'Stick', 'Transmission',
       'Hydraulics', 'Tire_Size', 'Track_Type',
       'Grouser_Type', 'Blade_Type', 'Steering_Controls']]

    #Year Made
    #setting value of 1000 to missing value
    interested_features['YearMade']=interested_features['YearMade'].replace(1000,np.nan)

    #fig, ax=plt.subplots(1,1)
    #ax.hist(interested_features['YearMade'],bins=100)

    #Machine hours
    machine_hours_dummy=interested_features['MachineHoursCurrentMeter'].where(interested_features['MachineHoursCurrentMeter']==0,1)

    X=machine_hours_dummy

    #Usage
    dummies_usage = pd.get_dummies(interested_features['UsageBand'])

    X = pd.concat([dummies_usage, X], axis=1, join_axes=[X.index])

    #Created dummy variables for Hydraulics
    dummies_hydraulics = pd.get_dummies(interested_features['Hydraulics'])
    #dummies_hydraulics.sum().sort_values()

    X = pd.concat([dummies_hydraulics['2 Valve'],dummies_hydraulics['Standard'],dummies_hydraulics['Auxiliary'], X], axis=1, join_axes=[X.index])

    #Created dummy variables for Enclosure
    dummies_enclosure = pd.get_dummies(interested_features['Enclosure'])
    #dummies_enclosure.sum().sort_values()

    X = pd.concat([dummies_enclosure['EROPS w AC'],dummies_enclosure['OROPS'], X], axis=1, join_axes=[X.index])

    #Created dummy variables for State
    dummies_state = pd.get_dummies(interested_features['state'], drop_first=True)
    #dummies_state.sum().sort_values()

    X = pd.concat([dummies_state['Florida'],dummies_state['Texas'],dummies_state['California'], X], axis=1, join_axes=[X.index])

    #create variable for the year of the tracktor sale
    interested_features['YearSale']=pd.to_datetime(interested_features['saledate']).dt.year
    #fig, ax=plt.subplots(1,1)
    #ax.hist(interested_features['YearSale'])

    #Create variable for the age of the vehicle at the time of sale
    interested_features['Age'] = interested_features['YearSale']-interested_features['YearMade']
    #fig, ax=plt.subplots(1,1)
    #ax.hist(interested_features['Age'])

    #Display rows with negative age
    wrong_age=interested_features[interested_features['YearMade']>interested_features['YearSale']]
    #Cap negative values at zero
    interested_features['Age']=interested_features['Age'].where(interested_features['Age']>0,0)
    #fig, ax=plt.subplots(1,1)
    #ax.hist(interested_features['Age'])
    interested_features['Age_sq']=interested_features['Age']**2

    X = pd.concat([interested_features['Age'],interested_features['Age_sq'], X], axis=1, join_axes=[X.index])

    #Dummies for product size
    dummies_productSize = pd.get_dummies(interested_features['ProductSize'])
    #dummies_productSize.sum().sort_values()

    X = pd.concat([dummies_productSize, X], axis=1, join_axes=[X.index])

    #Dummies for product class
    dummies_productClass = pd.get_dummies(interested_features['fiProductClassDesc'])
    interested_features['product_class_Backhoe_Loader'] = dummies_productClass['Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth']
    #dummies_productClass.sum().sort_values()

    X = pd.concat([interested_features['product_class_Backhoe_Loader'], X], axis=1, join_axes=[X.index])

    #Dummies product group
    dummies_productGroup = pd.get_dummies(interested_features['ProductGroup'])
    #dummies_productGroup.sum().sort_values()

    #X = pd.concat([dummies_productGroup, X], axis=1, join_axes=[X.index])

    #Dummies drive system
    dummies_driveSystem = pd.get_dummies(interested_features['Drive_System'])
    #dummies_driveSystem.sum().sort_values()

    X = pd.concat([dummies_driveSystem, X], axis=1, join_axes=[X.index])

    #Dummies stick ???
    dummies_stick = pd.get_dummies(interested_features['Stick'])
    #dummies_stick.sum().sort_values()

    X = pd.concat([dummies_stick, X], axis=1, join_axes=[X.index])

    #Dummies transmission
    dummies_transmission = pd.get_dummies(interested_features['Transmission'])
    #dummies_transmission.sum().sort_values()

    X = pd.concat([dummies_transmission['Standard'], X], axis=1, join_axes=[X.index])

    #Dummies tire size???
    dummies_tireSize = pd.get_dummies(interested_features['Tire_Size'])
    #dummies_tireSize.sum().sort_values()

    X = pd.concat([dummies_tireSize['20.5'], X], axis=1, join_axes=[X.index])

    #Dummies track type
    dummies_trackType = pd.get_dummies(interested_features['Track_Type'])
    #dummies_trackType.sum().sort_values()

    X = pd.concat([dummies_trackType, X], axis=1, join_axes=[X.index])

    #Dummies Grouser_Type
    dummies_Grouser_Type = pd.get_dummies(interested_features['Grouser_Type'])
    #dummies_Grouser_Type.sum().sort_values()
    #X = pd.concat([dummies_Grouser_Type['Double'],dummies_Grouser_Type['Triple'], X], axis=1, join_axes=[X.index])

    #Dummies Blade_Type???
    dummies_Blade_Type = pd.get_dummies(interested_features['Blade_Type'])
    dummies_Blade_Type.sum().sort_values()

    X = pd.concat([dummies_Blade_Type['PAT'], X], axis=1, join_axes=[X.index])

    #Dummies Steering control
    dummies_Steering_Controls = pd.get_dummies(interested_features['Steering_Controls'])
    #dummies_Steering_Controls.sum().sort_values()

    X = pd.concat([dummies_Steering_Controls['Conventional'], X], axis=1, join_axes=[X.index])

    return X

X=features_func(Train)
#A=X.describe()
y = Train['SalePrice']

#Model
def linear_model(X,y):
    model = LinearRegression()
    model.fit(X,y)
    y_predicted = model.predict(X)
    mse=mean_squared_error(y,y_predicted)
    mse_cv=(cross_val_score(model,X,y,cv=5,scoring='neg_mean_squared_error'))*-1
    #Residual scatter plot
    plt.scatter(y_predicted,y-y_predicted)
    return y_predicted, mse, mse_cv, model

y_predicted, mse, mse_cv, model=linear_model(X,np.log(y))

#Model summary
def model_summary(X,y):
    C = sm.add_constant(range(1,len(y)+1))
    C=pd.DataFrame(C)
    X = pd.concat([C.iloc[:,0],X], axis=1, join_axes=[X.index])

    model = sm.OLS(y,X)
    result=model.fit()
    return result.summary()

model_summary(X,np.log(y))

### Test dataset---------------------------------------------

X_test=features_func(Test)

Test['SalePrice'] = model.predict(X_test)
Test['SalePrice']=np.exp(Test['SalePrice'])
Test[['SalesID','SalePrice']].to_csv('data/output.csv', sep=',', index = False)

output = pd.read_csv('data/output.csv')
