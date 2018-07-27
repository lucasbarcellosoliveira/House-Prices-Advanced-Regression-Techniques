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
#train.drop(train[train["GrLivArea"]>4000].index,inplace=True)
test=complete.iloc[1460:,:]

outlier_idx=[4,11,13,20,46,66,70,167,178,185,199,224,261,309,313,318,349,412,423,440,454,477,478,523,540,581,588,595,654,688,691,774,798,875,898,926,970,987,1027,1109,1169,1182,1239,1256,1298,1324,1353,1359,1405,1442,1447]
train.drop(train.index[outlier_idx],inplace=True)

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

from sklearn import svm

model=svm.NuSVR(C=1350000,nu=0.74)

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
