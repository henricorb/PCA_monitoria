# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 08:20:17 2020

@author: Henrico
"""

import numpy as np
from matplotlib import pyplot as plt

def twoGaussian(N=200,sig1=0.4,sig2=0.4):
    # Função para geração de duas classes Gaussianas
    mu1 = np.array([2,2])
    cov1 = (sig1**2)*np.eye(2)
    mu2 = np.array([4,4])
    cov2 = (sig2**2)*np.eye(2)
    X1 = np.random.multivariate_normal(mu1, cov1, size=N)
    X2 = np.random.multivariate_normal(mu2, cov2, size=N)

    X = np.vstack((X1,X2))
    y = np.hstack((np.ones(N),np.zeros(N)))
    
    return X, y

# Função para geração da base de dados sintética
def twospirals(n_points, n_turns, ts=np.pi, tinc=1, noise=.3):
    """
     Returns the two spirals dataset.
     modificado de: 
         https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
    Primeiro gera uma espiral e obtem a segunda espelhando a primeira
    """
    # equação da espiral (coord polares): r = tinc*theta
    # n_points: número de pontos de cada espiral
    # n_turns: número de voltas das espirais
    # ts: ângulo inicial da espiral em radianos
    # tinc: taxa de crescimento do raio em função do ângulo
    # noise: desvio-padrão do ruído
    
    # Sorteando aleatoriamente pontos da espiral
    n = np.sqrt(np.random.rand(n_points,1))  #intervalo [0,1] equivale a [0,theta_max]
                                             #tomar a raiz quadrada ajuda a 
                                             #distribuir melhor os pontos
    ns = (ts)/(2*np.pi*n_turns) #ponto do intervalo equivalente a ts radianos
    n = ns + n_turns*n # intervalo [ns,ns+n_turns] equivalente a [ts, theta_max]
    n = n*(2*np.pi) #intervalo [ts, theta_max]
    
    # Espiral 1
    d1x = np.cos(n)*tinc*n + np.random.randn(n_points,1) * noise
    d1y = np.sin(n)*tinc*n + np.random.randn(n_points,1) * noise
    
    # Espiral 2
    d2x = -np.cos(n)*tinc*n + np.random.randn(n_points,1) * noise
    d2y = -np.sin(n)*tinc*n + np.random.randn(n_points,1) * noise
    
    spirals_points = np.vstack((np.hstack((d1x,d1y)),np.hstack((d2x,d2y))))
    points_labels = np.hstack((np.ones(n_points),np.zeros(n_points)))
    return (spirals_points, points_labels)

def synthData_plot(X,y):
    plt.plot(X[y==1,0], X[y==1,1], '.b', label='Classe 1')
    plt.plot(X[y==0,0], X[y==0,1], '.r', label='Classe 2')
    plt.legend(loc='upper right')
    plt.show()

# -----------------------------------------------------------------------------
def main():
    """ ======================================================================= """ 
    """ Script Principal """ 
    np.random.seed(39) #sankyuu
    #-----------------------------------------------------------------------------
    # Gerando as duas espirais para treino e teste
    # Espirais com duas voltas, angulo inicial pi rad, ruido 0.3.
    #-----------------------------------------------------------------------------
    Ntrain = 500
    Ntest = 300
    voltas = 2
    theta0 = np.pi
    noise=0.3
    
    X, y = twospirals(Ntrain,voltas,ts=theta0,tinc=1,noise=noise)
    plt.figure()
    plt.title('Treinamento')
    synthData_plot(X, y)
    
    Xtest, ytest = twospirals(Ntest,voltas,ts=theta0,tinc=1,noise=noise)
    plt.figure()
    plt.title('Teste')
    synthData_plot(Xtest, ytest)
    
    
if __name__ == "__main__":
    main()