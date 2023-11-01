# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:22:29 2020

@author: Henrico
"""

import numpy as np
from matplotlib import pyplot as plt
import mySynthData
import myPCA

""" Geração de Dados """

X,y = mySynthData.twospirals(200, 2, ts=np.pi, tinc=1, noise=.3)
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
Zstar,D,P,Dratio = myPCA.myPCA(Z)


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





