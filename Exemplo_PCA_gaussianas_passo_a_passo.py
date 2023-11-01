# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:22:29 2020

@author: Henrico
"""

import numpy as np
from matplotlib import pyplot as plt
import mySynthData
import myPCA

def twoGaussian(N=200,mu1 = np.array([2,2]),mu2 = np.array([4,4]),
                            cov1 = (0.4**2)*np.eye(2), cov2 = (0.4**2)*np.eye(2)):
    # Função para geração de duas classes Gaussianas
    X1 = np.random.multivariate_normal(mu1, cov1, size=N)
    X2 = np.random.multivariate_normal(mu2, cov2, size=N)

    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(N),np.zeros(N)))
    
    return X, y


""" Geração de Dados """
plt.close('all')
mu1=np.array([2,2])
mu2=np.array([5,5])
cov1 = (0.5**2)*np.eye(2)
cov2 = (0.5**2)*np.eye(2)

# Dados que podem dar problema
# mu1=np.array([2,2])
# mu2=np.array([5,5])
# rho=0.8 
# sig1=0.5 
# sig2=0.5
# cov1 = np.array([[sig1**2,rho*sig1*sig2],[rho*sig1*sig2,sig2**2]])
# cov2 = np.array([[.05**2,0],[0,.05**2]])

(X,y) = twoGaussian(200,mu1,mu2,cov1,cov2)
plt.figure()
mySynthData.synthData_plot(X, y)
plt.show()


""" Implementação de PCA """
#1)=================== Pré-tratamento ========================================
#1.1) Remover média coluna a coluna
Z = X - np.mean(X,0)
plt.figure()
plt.plot(Z[:,0],Z[:,1],'.')
plt.grid()

#1.2) Normalizar o desvio-padrão
Z = Z/np.std(Z,0,ddof=1)
plt.figure()
plt.plot(Z[:,0],Z[:,1],'.')
plt.grid()

#2)=================== Componentes Principais =================================
Zcov = np.matmul(Z.transpose(),Z) #Matriz de covariância
#2.1) Autovalores e autovetores da matriz de covariancia
# D - vetor com os autovalores
# P - matriz com os autovetores associados
(D,P) = np.linalg.eig(Zcov)
print(D)
print(P)

# Ordenando em ordem decrescente de autovalores
ind = np.flipud(np.argsort(D))
D = D[ind]
P = P[:,ind]

#2.2) Transformando os dados e proporção das componentes
Dratio = D/D.sum() #"Importância" de cada componente
Zstar = np.matmul(Z,P)

#2.3) Indicando a transformação
plt.figure()
plt.plot(Z[:,0],Z[:,1],'.') #Dados originais
plt.arrow(0,0,P[0,0],P[1,0]) #CP1
plt.arrow(0,0,P[0,1],P[1,1]) #CP2
plt.grid()

#2.4) Exibindo dados transformados
plt.figure()
plt.plot(Zstar[:,0],Zstar[:,1],'o')
plt.xlabel('CP1 ('+"{:.1f}".format(100*Dratio[0])+'%)')
plt.ylabel('CP2 ('+"{:.1f}".format(100*Dratio[1])+'%)')


#2.5) Plotando círculo de correlação
#dim1 vs PC1
rxi = np.corrcoef(Z[:,0], Zstar[:,0])[0,1]
rxj = np.corrcoef(Z[:,0], Zstar[:,1])[0,1]

#dim2 vs PC2
ryi = np.corrcoef(Z[:,1], Zstar[:,0])[0,1]
ryj = np.corrcoef(Z[:,1], Zstar[:,1])[0,1]

ri = [rxi,ryi]
rj = [rxj,ryj]

plt.figure()
# Coordenadas obtidas a partir das correlacoes
plt.arrow(0,0,rxi,rxj) #CP1
plt.arrow(0,0,ryi,ryj) #CP2
plt.xlim((-1.2,1.2))
plt.ylim((-1.2,1.2))

# Circulo para referencia
theta=np.linspace(0,2*np.pi,100)
a=np.cos(theta)
b=np.sin(theta)
plt.plot(a,b)
plt.grid()
plt.show()



myPCA.PCA_circle(Z,Zstar,Dratio)




