import pandas as pd

train=pd.read_csv("./all/train.csv")

train.drop(columns=["Id"],inplace=True)

#train.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)

train=pd.get_dummies(train,drop_first=True)

#train.dropna(inplace=True)
train.fillna(0,inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])

#from scipy.stats import skew
#numeric_feats=train.dtypes[train.dtypes != "object"].index
#skewed_feats=train[numeric_feats].apply(lambda x: skew(x.dropna()))
#skewed_feats=skewed_feats[skewed_feats > 0.75]
#skewed_feats=skewed_feats.index
#train.drop(columns=skewed_feats,inplace=True)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))

from sklearn import ensemble
from sklearn.model_selection import cross_val_score

for i in range(75,126):
    model=ensemble.GradientBoostingRegressor(n_estimators=i,criterion="mse")
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print(str(i)+"*GradientBoosted - R2:")
    print(scores.mean())
    print(str(i)+"*GradientBoosted - Desvio Padrão:")
    print(scores.std())

### Melhor: 110

model=ensemble.GradientBoostingRegressor(n_estimators=110,criterion="mse")

scores=cross_val_score(model,X,Y,cv=10)

print("*GradientBoosted - R2:")
print(scores.mean())
print("*GradientBoosted - Desvio Padrão:")
print(scores.std())

#%% RMSE

#RMSE=np.sqrt(-cross_val_score(model,X,Y.values,scoring="neg_mean_squared_error",cv=2)) #.values necessario para evitar erro de valor (nao finito)
#print("RMSE:")
#print(RMSE.mean())
#print("Desvio padrao:")
#print(RMSE.std())
