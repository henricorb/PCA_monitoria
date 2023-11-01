# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:22:29 2020

@author: Henrico
Implementação do tutorial disponível em
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import myPCA


""" Leitura de Dados """
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

""" Pré-processamento dos dados """
# 1) Método mais "correto", usando dataframe
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

""" Cálculo das Componentes Principais """
pca = PCA(n_components=2) # Indica que conservo apenas duas
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

""" Visualização dos dados """
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

""" ========================================================================="""
""" Refazendo no braço """

""" Pré-processamento dos dados """
x=np.array(df)[:,0:4]
x=x.astype('float')
x = x - np.mean(x,0)
x = x/np.std(x,0,ddof=1)
y = np.array(df)[:,4]


""" Cálculo das Componentes Principais """
Zstar,D,P,Dratio = myPCA.myPCA(x)

""" Visualização dos dados """
ind0 = np.where(y == 'Iris-setosa')
ind1 = np.where(y == 'Iris-versicolor')
ind2 = np.where(y == 'Iris-virginica')

plt.figure()
plt.plot(Zstar[ind0,0],Zstar[ind0,1],'ro')
plt.plot(Zstar[ind1,0],Zstar[ind1,1],'go')
plt.plot(Zstar[ind2,0],Zstar[ind2,1],'bo')
plt.xlabel('CP1 ('+"{:.1f}".format(100*Dratio[0])+'%)')
plt.ylabel('CP2 ('+"{:.1f}".format(100*Dratio[1])+'%)')

#2.5) Plotando círculo de correlação
myPCA.PCA_circle(x,Zstar,Dratio)



import seaborn as sns; sns.set(style="ticks", color_codes=True)

iris = sns.load_dataset("iris")

g = sns.pairplot(iris, hue="species")
