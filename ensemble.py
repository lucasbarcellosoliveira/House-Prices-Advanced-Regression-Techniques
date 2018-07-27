import pandas as pd

train=pd.read_csv("./all/train.csv")

train.drop(columns=["Id"],inplace=True)

#train.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)

train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

train=pd.get_dummies(train,drop_first=True)

#train.dropna(inplace=True)
train.fillna(0,inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

#%% GradientBoosted
model=ensemble.GradientBoostingRegressor(n_estimators=110,criterion="mse")

scores=cross_val_score(model,X,Y,cv=10)

print("*GradientBoosted - R2:")
print(scores.mean())
print("*GradientBoosted - Desvio Padrão:")
print(scores.std())

#%% RandomForest
model=ensemble.RandomForestRegressor(n_estimators=110)

scores=cross_val_score(model,X,Y,cv=10)

print("\nRandomForest - R2:")
print(scores.mean())
print("RandomForest - Desvio Padrão:")
print(scores.std())

#%% AdaBoost
model=ensemble.AdaBoostRegressor(n_estimators=110)

scores=cross_val_score(model,X,Y,cv=10)

print("\nAdaBoost - R2:")
print(scores.mean())
print("AdaBoost - Desvio Padrão:")
print(scores.std())

#%% Bagging
model=ensemble.BaggingRegressor(n_estimators=110)

scores=cross_val_score(model,X,Y,cv=10)

print("\nBagging - R2:")
print(scores.mean())
print("Bagging - Desvio Padrão:")
print(scores.std())

#%% ExtraTrees
model=ensemble.ExtraTreesRegressor(n_estimators=110)

scores=cross_val_score(model,X,Y,cv=10)

print("\nExtraTrees - R2:")
print(scores.mean())
print("ExtraTrees - Desvio Padrão:")
print(scores.std())
