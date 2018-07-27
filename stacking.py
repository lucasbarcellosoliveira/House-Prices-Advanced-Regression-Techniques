import pandas as pd

train=pd.read_csv("./all/train.csv")

test=pd.read_csv("./all/test.csv")

complete=pd.concat((train,test))

#complete.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)
#train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

res=complete.iloc[1460:,:]
res=res.loc[:,["Id","SalePrice"]]

complete.drop(columns=["Id"],inplace=True)

#numeric_feats = complete.dtypes[complete.dtypes != "object"].index
#
#from scipy.stats import skew
#
#skewed_feats = complete[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#skewness = pd.DataFrame({'Skew' :skewed_feats})
#    
#skewness = skewness[abs(skewness) > 0.75]
#
#skewness.dropna(inplace=True)
#
#from scipy.special import boxcox1p
#skewed_features = skewness.index
#lam = 0.15
#for feat in skewed_features:
#    complete[feat] = boxcox1p(complete[feat], lam)

#complete.fillna(0,inplace=True)

varNone=["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType",'MSSubClass']

for i in varNone:
    complete[i] = complete[i].fillna('None')

complete["LotFrontage"] = complete.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

var0=['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath',
      'BsmtHalfBath',"MasVnrArea"]

for i in var0:
    complete[i] = complete[i].fillna(0)

varMode=['Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType']

for i in varMode:
    complete[i] = complete[i].fillna(complete[i].mode()[0])

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(complete[c].values)) 
    complete[c] = lbl.transform(list(complete[c].values))

complete=pd.get_dummies(complete,drop_first=True)

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

#from sklearn.preprocessing import Imputer
#imp=Imputer(axis=1,strategy="mean")
#X=imp.fit_transform(X)
#test=imp.fit_transform(test)

#from sklearn.preprocessing import PolynomialFeatures

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

#models=[
#    RandomForestRegressor(n_estimators=110),
#    ExtraTreesRegressor(n_estimators=110),
#    GradientBoostingRegressor(n_estimators=110),
#    LassoCV(alphas=[56.0],fit_intercept=False,max_iter=1000000),
#    RidgeCV(alphas=[15.0],fit_intercept=False),
#    XGBRegressor(n_estimators=650),
#    #XGBRegressor(n_estimators=1100,max_depth=2),
#    #KNeighborsRegressor(n_neighbors=6,weights="distance",p=1),
#    SVR(C=1350000,epsilon=1200),
#    NuSVR(C=1350000,nu=0.74)
#]

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso

models=[
        Lasso(alpha =0.0005),
        ElasticNet(alpha=0.0005, l1_ratio=.9),
        KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
        GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber'),
        XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             nthread = -1)
]

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

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
import numpy as np
from sklearn.model_selection import KFold
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,models,meta,n):
        self.models=models
        self.meta=meta
        self.n=n
    def fit(self, X, y):
        self.models2 = [list() for x in self.models]
        self.meta2 = clone(self.meta)
        kfold = KFold(n_splits=self.n,shuffle=True)
        others=np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            for trainidx, holdout in kfold.split(X, y):
                self.models2[i].append(clone(model))
                instance.fit(X[trainidx],y[trainidx])
                others[holdout, i]=instance.predict(X[holdout])
        self.meta2.fit(others,y)
        return self
    def predict(self, X):
        meta_features=np.column_stack([
            np.column_stack([model.predict(X) for model in models]).mean(axis=1)
            for models in self.models2])
        return self.meta2.predict(meta_features)

model=StackingAveragedModels(models,RidgeCV(),10)

train["SalePrice"] = np.log1p(train["SalePrice"])

#%% cross_val_score

#scores=cross_val_score(model,X,Y.values,cv=10)
#
#print("R2:")
#print(scores.mean())
#print("Desvio Padrão:")
#print(scores.std())

#%% RMSE

#RMSE=np.sqrt(-cross_val_score(model,X,Y.values,scoring="neg_mean_squared_error",cv=2)) #.values necessario para evitar erro de valor (nao finito)
#print("RMSE:")
#print(RMSE.mean())
#print("Desvio padrao:")
#print(RMSE.std())

#%% CSV

model.fit(X,Y)

pred=model.predict(test)

pred=np.expm1(pred)

print("Escrevendo em arquivo")
res["SalePrice"]=pred
res.to_csv("res.csv",index=False)
