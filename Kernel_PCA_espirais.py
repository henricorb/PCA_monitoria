# -*- coding: utf-8 -*-
"""
@author: Henrico

"""

import numpy as np
from matplotlib import pyplot as plt
import myPCA
import mySynthData
from sklearn.decomposition import KernelPCA

""" Geração dos dados """
def twoGaussian(N=100,mu1 = np.array([2,2]),mu2 = np.array([4,4]),
                            cov1 = (0.4**2)*np.eye(2), cov2 = (0.4**2)*np.eye(2)):
    # Função para geração de duas classes Gaussianas
    X1 = np.random.multivariate_normal(mu1, cov1, size=N)
    X2 = np.random.multivariate_normal(mu2, cov2, size=N)

    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(N),np.zeros(N)))
    
    return X, y

def myKernel(xi, xj, h):
    # Entradas
    #   xi: vetor de entrada, array (n,)
    #   xj: vetor de centro, array (n,)
    #   h: escalar, abertura do kernel
    # Saídas
    #   fx: escalar, resultado do kernel
    
    xv=(xi-xj)/h
    xv2 = np.sum(xv**2)
    fx = np.exp(-0.5*xv2)
    fx=np.float64(fx)
    
    return fx

def kernelMat(X,h):
    # Calcula Matriz de Kernels  
    N = X.shape[0]
    kMat = np.zeros((N,N),float)
    for i in range(N):
        xi = X[i,]
        for j in range (N):
            xj = X[j,]
            kMat[i,j] = myKernel(xi, xj, h)
    return kMat


""" Geração de Dados """
X,y = mySynthData.twospirals(200, 2, ts=np.pi, tinc=1, noise=.3)
plt.figure()
mySynthData.synthData_plot(X, y)
plt.show()



""" Computando a matriz de Kernel e 'centralizando' """
N = X.shape[0]
h=0.5
K = kernelMat(X,h)
N_1 = (1/N)*np.eye(N)
Kc = (K - np.matmul(N_1,K) - np.matmul(K,N_1) 
    + np.matmul(np.matmul(N_1,K), N_1) )

""" Achando componentes principais """
(D,P) = np.linalg.eig(Kc)
ind = np.flipud(np.argsort(D))
D = D[ind]
P = P[:,ind]
Dratio = D/D.sum() #"Importância" de cada componente

""" Transformando os dados """
Zstar=np.zeros((N,N))
for d in range(0,N): #para cada amostra
    for i in range(0,N): #para cada dimensão da CP
        S = 0
        for j in range(0,N): # para cada CP
            S = S + P[i,j]*Kc[d,j]
        Zstar[d,i] = S


""" Visualização dos dados """
plt.figure()
plt.plot(Zstar[0:50,0],Zstar[0:50,1],'ro')
plt.plot(Zstar[50:100,0],Zstar[50:100,1],'bo')
plt.xlabel('CP1 ('+"{:.1f}".format(100*Dratio[0])+'%)')
plt.ylabel('CP2 ('+"{:.1f}".format(100*Dratio[1])+'%)')


""" Com bibliotecas """
transformer = KernelPCA(n_components=2, kernel='rbf')
X_transformed = transformer.fit_transform(X)
X_transformed.shape
plt.figure()
plt.plot(X_transformed[0:50,0],X_transformed[0:50,1],'ro')
plt.plot(X_transformed[50:100,0],X_transformed[50:100,1],'bo')

myPCA.PCA_circle(X,X_transformed,[0,0])



