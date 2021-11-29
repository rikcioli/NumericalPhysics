# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:12:49 2021

@author: rikci
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

N = 20
N_cor = 20
N_cf = 1000
eps = 1.4
a = 0.5


def S(j,x): # harm. osc. S
    jp = (j+1)%N # next site
    jm = (j-1)%N # previous site
    return a*x[j]**2/2 + x[j]*(x[j] - x[jp] - x[jm])/a

def update(x):
    for j in range(0,N):
        old_x = x[j] # save original value
        old_Sj = S(j,x)
        x[j] = x[j] + rnd.uniform(-eps,eps) # update x[j]
        dS = S(j,x) - old_Sj # change in action
        if dS>0 and np.exp(-dS)<rnd.uniform(0,1):
            x[j] = old_x # restore old value
            
def compute_G(x,n):
    g = 0
    for j in range(0,N):
        g = g + x[j]*x[(j+n)%N]
    return g/N

def MCaverage(x,G):
    for j in range(0,N): # initialize x
        x[j] = 0
    for j in range(0,5*N_cor): # thermalize x
        update(x)
    for alpha in range(0,N_cf): # loop on random paths
        for j in range(0,N_cor):
            update(x)
        for n in range(0,N):
            G[alpha][n] = compute_G(x,n)
    collect = np.zeros(N)
    for n in range(0,N): # compute MC averages
        avg_G = 0
        for alpha in range(0,N_cf):
            avg_G = avg_G + G[alpha][n]
        avg_G = avg_G/N_cf
        collect[n] = avg_G
        print("G(%d) = %g" % (n,avg_G))
    deltaE = np.zeros(N)
    t = np.linspace(0, N*a, N, endpoint=False)
    for n in range(N):
        deltaE[n] = np.log(np.abs(collect[n]/collect[(n+1)%N]))/a
    plt.plot(t, deltaE, 'g^')
    plt.plot(t, np.ones(N))
    plt.xlabel("t")
    plt.ylabel("DeltaE(t)")
    
x = np.zeros(N)
G = np.zeros((N_cf, N))
MCaverage(x,G)