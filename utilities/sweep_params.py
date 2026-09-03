#!/usr/bin/env python3
import os
import numpy as np

#%%

R = 0.4699/2
N = 5
rho = 1.225
sos = 343
AR = 7

Mt_min = 0.3
Mt_max = 0.8
Mt = np.arange(N)/(N-1)*(Mt_max-Mt_min)+Mt_min
omega = Mt*sos/R

CT_min = 0.004
CT_max = 0.015
CT = np.arange(N)/(N-1)*(CT_max-CT_min)+CT_min

delta_min = 0.25
delta_max = 2
delta = (np.arange(N)/(N-1)*(delta_max-delta_min)+delta_min)*(R/AR)

r_min = 0.25
r_max = 3
r = (np.arange(N)/(N-1)*(r_max-r_min)+r_min)*(R/AR)
c = r/R

sep_distance = (delta+c[:,None]*R/2)


A_min = np.pi/4*(R/2)**2
A_max = np.pi/4*(R*4)**2
A = np.arange(N)/(N-1)*(A_max-A_min)+A_min
R = np.sqrt(A/np.pi)

DL = CT*rho*(Mt[:,None]*sos)**2



DL_min = 100
DL_max = 500
DL = np.arange(N)/(N-1)*(DL_max-DL_min)+DL_min

CT = DL[:,None]/(rho*(Mt*sos)**2)

A_min = np.pi/4*(R/2)**2
A_max = np.pi/4*(R*4)**2
A = np.arange(N)/(N-1)*(A_max-A_min)+A_min
R = np.sqrt(A/np.pi)

T_min = np.round((DL[:,None]*A).min())
T_max = (DL[:,None]*A).max()
T = np.arange(N)/(N-1)*(T_max-T_min)+T_min
