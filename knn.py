import pandas as pd

train=pd.read_csv("./all/train.csv")

train.drop(columns=["Id"],inplace=True)

#train.drop(columns=["LotFrontage","MasVnrArea","GarageYrBlt"],inplace=True)

train=pd.get_dummies(train,drop_first=True)

#train.dropna(inplace=True)
train.fillna(0,inplace=True)

Y=train.loc[:,"SalePrice"]
X=train.drop(columns=["SalePrice"])

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X=scaler.fit_transform(X)
#Y=scaler.fit_transform(Y.values.reshape(-1,1))

from sklearn import neighbors
from sklearn.model_selection import cross_val_score

#%% kNN - Peso Uniforme - Distância Euclidiana
for i in range(1,21):
    model=neighbors.KNeighborsRegressor(n_neighbors=i,p=1)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print("\n"+str(i)+"-kNN - R2:")
    print(scores.mean())
    print(str(i)+"-kNN - Desvio Padrão:")
    print(scores.std())

#%% Melhor kNN - Peso Uniforme - Distância Euclidiana: k=8
model=neighbors.KNeighborsRegressor(n_neighbors=8)

scores=cross_val_score(model,X,Y,cv=10)

print("\n8-kNN - R2:")
print(scores.mean())
print("8-kNN - Desvio Padrão:")
print(scores.std())

#%% kNN - Peso Ponderado - Distância Euclidiana
for i in range(1,21):
    model=neighbors.KNeighborsRegressor(n_neighbors=i,weights="distance",p=2)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print("\n"+str(i)+"-kNN - R2:")
    print(scores.mean())
    print(str(i)+"-kNN - Desvio Padrão:")
    print(scores.std())

#%% Melhor kNN - Peso Ponderado - Distância Euclidiana: k=8
model=neighbors.KNeighborsRegressor(n_neighbors=8,weights="distance")

scores=cross_val_score(model,X,Y,cv=10)

print("\n8-kNN - R2:")
print(scores.mean())
print("8-kNN - Desvio Padrão:")
print(scores.std())

#%% kNN - Peso Uniforme - Distância de Manhattan
for i in range(1,21):
    model=neighbors.KNeighborsRegressor(n_neighbors=i,p=1)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print("\n"+str(i)+"-kNN - R2:")
    print(scores.mean())
    print(str(i)+"-kNN - Desvio Padrão:")
    print(scores.std())

#%% Melhor kNN - Distância Uniforme - Distância de Manhattan: k=6
model=neighbors.KNeighborsRegressor(n_neighbors=6,p=1)

scores=cross_val_score(model,X,Y,cv=10)

print("\n6-kNN - R2:")
print(scores.mean())
print("6-kNN - Desvio Padrão:")
print(scores.std())

#%% kNN - Peso Ponderado - Distância de Manhattan
for i in range(1,21):
    model=neighbors.KNeighborsRegressor(n_neighbors=i,weights="distance",p=1)
    
    scores=cross_val_score(model,X,Y,cv=10)
    
    print("\n"+str(i)+"-kNN - R2:")
    print(scores.mean())
    print(str(i)+"-kNN - Desvio Padrão:")
    print(scores.std())

#%% Melhor kNN - Distância Ponderado - Distância de Manhattan: k=6
model=neighbors.KNeighborsRegressor(n_neighbors=6,weights="distance",p=1)

scores=cross_val_score(model,X,Y,cv=10)

print("\n6-kNN - R2:")
print(scores.mean())
print("6-kNN - Desvio Padrão:")
print(scores.std())
