#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------
#
# Este programa faz a leitura dos dados da boia em
# .csv filtradas pelo Joey
# 
# 21MAI2020 ultima atualização
#-------------------------------------------------
# Bibliotecas
#
import os,sys,time,datetime
from pathlib import Path
home = str(Path.home()) #define $HOME

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.io import loadmat #carrega m-files




def matlab2datetime(matlab_datenum):
    import datetime as dt
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac



#-----
#Dados

boia=pd.read_csv(str(home)+'/Dropbox/projetos/antartica/insitu/spot0335/brutos_op38/Wave_Data_filtered2.csv', encoding='ISO-8859-1')

Hs=np.asarray(boia.Hs)
Dp=np.asarray(boia.Dir_p)
D=np.asarray(boia.Dir_m)
Tp=np.asarray(boia.Tp)
Tm01=np.asarray(boia.Tm01)
Tm02=np.asarray(boia.Tm02)

a1_f1=np.asarray(boia.a1_f1)   # sao 129 frequencias, entao tenho 129 E, 129 a1,129 a2
a2_f1=np.asarray(boia.a2_f1)   # 129 b1 e 129 b2. Tenho que extrair todos eles para
b1_f1=np.asarray(boia.b1_f1)   # calcular o espectro 2D a cada passo de tempo. 
b2_f1=np.asarray(boia.b2_f1)   # OBS: talvez fazer um loop aqui pra carregar


t=np.asarray(boia.time)                                  #carrega arquivo.csv passando pra np.array
Time = np.zeros((len(t)),dtype=int)                      #converte do formato de tempo matlab
Time=[matlab2datetime(t[i]) for i in range(0, len(t))]   #para datetime do python

    
   
#OBS
#Time eh uma lista mas Time[0] acessa o formato datetime.datetime, por isso
#Time[i].strftime funciona.


#-------------------------------------
#Printa dados para direcionar para asc


for i in range(0,len(t)):
    print(Time[i].strftime("%Y%m%d%H%M"), '% 1.2f'% Hs[i], '% 2.2f'% Tp[i],'% 3.d'% Dp[i])


#---
#FIM
quit()




