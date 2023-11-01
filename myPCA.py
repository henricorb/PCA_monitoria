# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:22:29 2020

@author: Henrico
"""

import numpy as np
from matplotlib import pyplot as plt

def myPCA(Z):
    #Entrada
    #   Z: matriz com os dados de entrada pré-processados (média coluna a coluna
    #      nula e possivelmente desvio-padrão unitário)
    #Saídas
    #   Zstar: dados transformados
    #   D: autovalores da matriz de covariância, ordenados do maior para o menor
    #   P: matriz com os vetores associados à cada componente principal
    #   Dratio: as cargas de cada componente principal, ordenados do menor para o maior
    
    Zcov = np.matmul(Z.transpose(),Z) #Matriz de covariância
    #2.1) Autovalores e autovetores da matriz de covariancia
    # D - vetor com os autovalores
    # P - matriz com os autovetores associados
    (D,P) = np.linalg.eig(Zcov)
    
    # Ordenando em ordem decrescente de autovalores
    ind = np.flipud(np.argsort(D))
    D = D[ind]
    P = P[:,ind]
    
    #2.2) Transformando os dados e proporção das componentes
    Dratio = D/D.sum() #"Importância" de cada componente
    Zstar = np.matmul(Z,P)
        
    return Zstar,D,P,Dratio

def PCA_circle(Z,Zstar,Dratio):
    PC1 = Zstar[:,0]
    PC2 = Zstar[:,1]
    
    ndim = Z.shape[1]
    
    #calculando as coordenadas de cada direção
    r=np.zeros((ndim,2))
    for i in range(0,ndim):
        r[i,0] = np.corrcoef(Z[:,i], PC1)[0,1]
        r[i,1] = np.corrcoef(Z[:,i], PC2)[0,1]
    
    #Plotando
    plt.figure()
    for i in range(0,ndim):
        plt.arrow(0,0,r[i,0],r[i,1])
        plt.text(r[i,0],r[i,1], 'Var'+str(i+1))
    
    # Circulo para referencia
    theta=np.linspace(0,2*np.pi,100)
    a=np.cos(theta)
    b=np.sin(theta)
    plt.plot(a,b)
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    plt.xlabel('CP1 ('+"{:.1f}".format(100*Dratio[0])+'%)')
    plt.ylabel('CP2 ('+"{:.1f}".format(100*Dratio[1])+'%)')
    plt.grid()
    plt.show()








