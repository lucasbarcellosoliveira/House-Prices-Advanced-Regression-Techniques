import pandas as pd

train=pd.read_csv("./all/train.csv")

train.drop(columns=["Id"],inplace=True)

#train.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)
#train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

#train.drop(columns=["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"])

outlier_idx=[4,11,13,20,46,66,70,167,178,185,199,224,261,309,313,318,349,412,423,440,454,477,478,523,540,581,588,595,654,688,691,774,798,875,898,926,970,987,1027,1109,1169,1182,1239,1256,1298,1324,1353,1359,1405,1442,1447]
train.drop(train.index[outlier_idx],inplace=True)

train=pd.get_dummies(train,drop_first=True)

#train.dropna(inplace=True)
train.fillna(0,inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])

#from scipy.stats import skew
#numeric_feats=train.dtypes[train.dtypes!="object"].index
#skewed_feats=train[numeric_feats].apply(lambda x: skew(x.dropna()))
#skewed_feats=skewed_feats[skewed_feats>0.75]
#skewed_feats=skewed_feats.index
#train.drop(columns=skewed_feats,inplace=True)

from sklearn.preprocessing import Imputer
imp=Imputer(axis=1,strategy="mean")
X=imp.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))

from sklearn import svm
from sklearn.model_selection import cross_val_score

#%% SVR
for i in range (1150,1251,10):
    model=svm.SVR(C=1350000,epsilon=i)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print(str(i)+"*SVR - R2:")
    print(scores.mean())
    print(str(i)+"*SVR - Desvio Padrão:")
    print(scores.std())

#%% Melhor SVR
model=svm.SVR(C=1350000,epsilon=1200)

scores=cross_val_score(model,X,Y,cv=10)

print("*SVR - R2:")
print(scores.mean())
print("*SVR - Desvio Padrão:")
print(scores.std())

#%% NuSVR
for i in range(65,76):
    model=svm.NuSVR(C=1350000,nu=i/100)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print(i/100)
    print("\nNuSVR - R2:")
    print(scores.mean())
    print("SVR - Desvio Padrão:")
    print(scores.std())

#%% Melhor NuSVR
model=svm.NuSVR(C=1350000,nu=0.74)

scores=cross_val_score(model,X,Y,cv=10)

print("\nNuSVR - R2:")
print(scores.mean())
print("SVR - Desvio Padrão:")
print(scores.std())

#%% LinearSVR
model=svm.LinearSVR(C=1350000)

scores=cross_val_score(model,X,Y,cv=10)

print("\nNuSVR - R2:")
print(scores.mean())
print("SVR - Desvio Padrão:")
print(scores.std())

#%% RMSE

#RMSE=np.sqrt(-cross_val_score(model,X,Y.values,scoring="neg_mean_squared_error",cv=2)) #.values necessario para evitar erro de valor (nao finito)
#print("RMSE:")
#print(RMSE.mean())
#print("Desvio padrao:")
#print(RMSE.std())
