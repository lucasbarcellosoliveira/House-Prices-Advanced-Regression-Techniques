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

#from scipy.stats import skew
#numeric_feats = train.dtypes[train.dtypes != "object"].index
#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#train.drop(columns=skewed_feats,inplace=True)

#from sklearn.preprocessing import Imputer
#imp=Imputer(axis=1,strategy="mean")
#X=imp.fit_transform(X)

from sklearn import tree
from sklearn.model_selection import cross_val_score

#%% Árvore de Decisão
model=tree.DecisionTreeRegressor()

scores=cross_val_score(model,X,Y,cv=10)

print("*R2:")
print(scores.mean())
print("*Desvio Padrão:")
print(scores.std())

#%% Árvore de Decisão - Critério MSE
model=tree.DecisionTreeRegressor(criterion="mse")

scores=cross_val_score(model,X,Y,cv=10)

print("\nmse - R2:")
print(scores.mean())
print("mse - Desvio Padrão:")
print(scores.std())

#%% Extra Tree
model=tree.ExtraTreeRegressor(criterion="mse")

scores=cross_val_score(model,X,Y,cv=10)

print("\nExtra/mse - R2:")
print(scores.mean())
print("Extra/mse - Desvio Padrão:")
print(scores.std())