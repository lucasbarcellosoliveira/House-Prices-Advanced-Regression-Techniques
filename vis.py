import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("./all/train.csv")

print(train["SalePrice"].describe())

cm=train.corr()
f,ax=plt.subplots(figsize=(12, 9))
sns.heatmap(cm,vmax=.8,square=True);

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()

X,Y=train.iloc[:,:80],train.iloc[:,80]

#pd.plotting.scatter_matrix(X)

import numpy as np

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(train['SalePrice'],kde=False)
plt.xlabel('Sale price')
plt.axis([0,800000,0,180])
plt.subplot(122)
sns.distplot(np.log(train['SalePrice']),kde=False)
plt.xlabel('Log (sale price)')
plt.axis([10,14,0,180])
