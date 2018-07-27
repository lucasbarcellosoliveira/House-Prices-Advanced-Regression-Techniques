import pandas as pd

train=pd.read_csv("./all/train.csv")

train.drop(columns=["Id"],inplace=True)

#train.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)
train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

#train.drop(columns=["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"])

#outlier_idx=[4,11,13,20,46,66,70,167,178,185,199,224,261,309,313,318,349,412,423,440,454,477,478,523,540,581,588,595,654,688,691,774,798,875,898,926,970,987,1027,1109,1169,1182,1239,1256,1298,1324,1353,1359,1405,1442,1447]
#train.drop(train.index[outlier_idx],inplace=True)

train=pd.get_dummies(train,drop_first=True)

#train.dropna(inplace=True)
#train.fillna(0,inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])

from scipy.stats import skew
numeric_feats=train.dtypes[train.dtypes != "object"].index
skewed_feats=train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats=skewed_feats[skewed_feats > 0.75]
skewed_feats=skewed_feats.index
train.drop(columns=skewed_feats,inplace=True)

from sklearn.preprocessing import Imputer
imp=Imputer(axis=1,strategy="mean")
X=imp.fit_transform(X)

#from sklearn.preprocessing import PolynomialFeatures
#
#poly=PolynomialFeatures(2)
#X=poly.fit_transform(X)

#from sklearn.decomposition import PCA
#
#pca=PCA(n_components=10)
#X=pca.fit_transform(X)

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#%% Lasso
model=linear_model.LassoCV(alphas=[56.0],fit_intercept=False,normalize=True)

scores=cross_val_score(model,X,Y,cv=10)

print("Lasso - R2:")
print(scores.mean())
print("Lasso - Desvio Padrão:")
print(scores.std())

#%% Ridge
model=linear_model.RidgeCV(alphas=[15.0],fit_intercept=False,normalize=True)

scores=cross_val_score(model,X,Y,cv=10)

print("\n*Ridge - R2:")
print(scores.mean())
print("*Ridge - Desvio Padrão:")
print(scores.std())

#%% BayesianRidge
model=linear_model.BayesianRidge(fit_intercept=False,normalize=True)

scores=cross_val_score(model,X,Y,cv=10)

print("\nBayesianRidge - R2:")
print(scores.mean())
print("BayesianRidge - Desvio Padrão:")
print(scores.std())

#%% Huber MELHOR COM OUTLIER EM 4000
model=linear_model.HuberRegressor(fit_intercept=False)

scores=cross_val_score(model,X,Y,cv=10)

print("\nHuber - R2:")
print(scores.mean())
print("Huber - Desvio Padrão:")
print(scores.std())

#%% RANSAC
model=linear_model.RANSACRegressor()

scores=cross_val_score(model,X,Y,cv=10)

print("\nRANSAC - R2:")
print(scores.mean())
print("RANSAC - Desvio Padrão:")
print(scores.std())