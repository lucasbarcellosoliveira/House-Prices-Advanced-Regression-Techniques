import pandas as pd

train=pd.read_csv("./all/train.csv")

test=pd.read_csv("./all/test.csv")

complete=pd.concat((train,test))

#complete.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)
#train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

res=complete.iloc[1460:,:]
res=res.loc[:,["Id","SalePrice"]]

complete.drop(columns=["Id"],inplace=True)

complete=pd.get_dummies(complete,drop_first=True)

#complete.fillna(0,inplace=True)

train=complete.iloc[:1460,:]
test=complete.iloc[1460:,:]

#outlier_idx=[4,11,13,20,46,66,70,167,178,185,199,224,261,309,313,318,349,412,423,440,454,477,478,523,540,581,588,595,654,688,691,774,798,875,898,926,970,987,1027,1109,1169,1182,1239,1256,1298,1324,1353,1359,1405,1442,1447]
#train.drop(train.index[outlier_idx],inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])
test=test.drop(columns=["SalePrice"])

#from scipy.stats import skew
#numeric_feats=train.dtypes[train.dtypes!="object"].index
#skewed_feats=train[numeric_feats].apply(lambda x: skew(x.dropna()))
#skewed_feats=skewed_feats[skewed_feats>0.75]
#skewed_feats=skewed_feats.index
#train.drop(columns=skewed_feats,inplace=True)

from sklearn.preprocessing import Imputer
imp=Imputer(axis=1,strategy="mean")
X=imp.fit_transform(X)
test=imp.fit_transform(test)

from sklearn.preprocessing import PolynomialFeatures

#poly=PolynomialFeatures(2)
#X=poly.fit_transform(X)
#test=poly.fit_transform(test)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))
test=scaler.fit_transform(test)

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from xgboost import XGBRegressor

models=[
    RandomForestRegressor(n_estimators=110),
    ExtraTreesRegressor(n_estimators=110),
    GradientBoostingRegressor(n_estimators=110),
    LassoCV(alphas=[56.0],fit_intercept=False,max_iter=1000000),
    RidgeCV(alphas=[15.0],fit_intercept=False),
    XGBRegressor(n_estimators=650),
    #XGBRegressor(n_estimators=1100,max_depth=2),
    #KNeighborsRegressor(n_neighbors=6,weights="distance",p=1),
    SVR(C=1350000,epsilon=1200),
    NuSVR(C=1350000,nu=0.74)
]

weights=[1,1,1,1,1,2,2,2]

#import numpy as np
#from sklearn.cross_validation import KFold
#
#folds=list(KFold(len(Y),n_folds=10,shuffle=True))
#S_train=np.zeros((X.shape[0],len(models)))
#S_test=np.zeros((test.shape[0],len(models))) 
#for i,reg in enumerate(models):
#    print ("Ajustando modelos")
#    S_test_i = np.zeros((test.shape[0],len(folds))) 
#    for j,(train_idx,test_idx) in enumerate(folds):
#        X_train=X[train_idx]
#        y_train=Y[train_idx]
#        X_holdout=X[test_idx]
#        reg.fit(X_train,y_train)
#        y_pred=reg.predict(X_holdout)[:]
#        S_train[test_idx,i]=y_pred
#        S_test_i[:,j]=reg.predict(test)[:]
#    S_test[:,i]=S_test_i.mean(1)

from sklearn.base import BaseEstimator, RegressorMixin
class MeuEnsemble(BaseEstimator,RegressorMixin):
    def __init__(self,models,weights=[]):
        self.models=models
        if not weights:
            self.weights=[1]*len(models)
        else:
            self.weights=weights
    def fit(self,X,Y):
        for i in self.models:
            print ("Ajustando modelos")
            i.fit(X,Y)
        return
    def predict(self,Y):
        pred=[]
        for i in Y:
            temp=0
            for j in range(len(self.models)):
                temp+=self.weights[j]*models[j].predict(i.reshape(1,-1))
            temp/=sum(self.weights)
            pred+=[temp]
        return pred

model=MeuEnsemble(models,weights)

#scores=cross_val_score(model,X,Y,cv=10)
#
#print("R2:")
#print(scores.mean())
#print("Desvio Padrão:")
#print(scores.std())

model.fit(X,Y)

pred=model.predict(test)

print("Escrevendo em arquivo")
res["SalePrice"]=[i[0] for i in pred] #para remover colchetes do csv
res.to_csv("res.csv",index=False)

#%% RMSE

#RMSE=np.sqrt(-cross_val_score(model,X,Y.values,scoring="neg_mean_squared_error",cv=2)) #.values necessario para evitar erro de valor (nao finito)
#print("RMSE:")
#print(RMSE.mean())
#print("Desvio padrao:")
#print(RMSE.std())

