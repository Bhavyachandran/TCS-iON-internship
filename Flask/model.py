# Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Loading dataset
df=pd.read_excel('df_Pro.xlsx')
# spliting of data into target and features
X=df.drop(['Employee_Name','EmpID','DOB','DateofHire', 'DateofTermination','LastPerformanceReview_Date','ManagerID',
           'YearofHire', 'Yearoflastperformence',],axis=1)
y=df['Salary']

# split into train and split test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBRegressor
xg_boost=XGBRegressor()
xg_boost = xg_boost.fit(X_train,y_train)
y_pred_xg_boost=xg_boost.predict(X_test)
print("R squared value  of XG Boost is : ",r2_score(y_test,y_pred_xg_boost))

# saving model using pickle
pickle.dump(xg_boost, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))