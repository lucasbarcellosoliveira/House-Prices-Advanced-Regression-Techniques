import pandas as pd

train=pd.read_csv("./all/train.csv")

test=pd.read_csv("./all/test.csv")

complete=pd.concat((train,test))

#complete.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)
train.drop(train[train["GrLivArea"]>3500].index,inplace=True)

res=complete.iloc[1460:,:]
res=res.loc[:,["Id","SalePrice"]]

complete.drop(columns=["Id"],inplace=True)

complete=pd.get_dummies(complete,drop_first=True)

#complete.fillna(0,inplace=True)

train=complete.iloc[:1460,:]
test=complete.iloc[1460:,:]

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])
test=test.drop(columns=["SalePrice"])

from sklearn.preprocessing import Imputer
imp=Imputer(axis=1,strategy="mean")
X=imp.fit_transform(X)
test=imp.fit_transform(test)

#from sklearn.preprocessing import PolynomialFeatures
#
#poly=PolynomialFeatures(2)
#X=poly.fit_transform(X)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))
test=scaler.fit_transform(test)

from sklearn import linear_model

#model=linear_model.RidgeCV(alphas=[15.0],fit_intercept=False)
model=linear_model.RidgeCV(fit_intercept=False)

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
