import pandas as pd

train=pd.read_csv("./all/train.csv")

test=pd.read_csv("./all/test.csv")

complete=pd.concat((train,test))

res=complete.iloc[1460:,:]
res=res.loc[:,["Id","SalePrice"]]

complete.drop(columns=["Id"],inplace=True)

complete=pd.get_dummies(complete,drop_first=True)

complete.fillna(0,inplace=True)

train=complete.iloc[:1460,:]
test=complete.iloc[1460:,:]

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])
test=test.drop(columns=["SalePrice"])

#from sklearn.preprocessing import PolynomialFeatures
#
#poly=PolynomialFeatures(2)
#X=poly.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))
test=scaler.fit_transform(test)

from sklearn import ensemble

model=ensemble.GradientBoostingRegressor(n_estimators=110,criterion="mse")

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,X,Y,cv=10)
print("\nR2:")
print(scores.mean())
print("Desvio Padrão:")
print(scores.std())

model.fit(X,Y)

pred=model.predict(test)

res["SalePrice"]=pred
res.to_csv("res.csv",index=False)

#%% RMSE

#RMSE=np.sqrt(-cross_val_score(model,X,Y.values,scoring="neg_mean_squared_error",cv=2)) #.values necessario para evitar erro de valor (nao finito)
#print("RMSE:")
#print(RMSE.mean())
#print("Desvio padrao:")
#print(RMSE.std())
