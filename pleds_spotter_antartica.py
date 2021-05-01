import time as timem
from numpy import *
from pylab import *
import numpy as np
import matplotlib.pylab as pl
from scipy.signal import lfilter, filtfilt, butter
# import espec
import os
import pandas as pd
from datetime import datetime
import matplotlib
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd

plt.close('all')

espe = pd.read_table('/home/hp/Documents/inpe/espe_201912.txt', sep=',', header=None)
dire = pd.read_table('/home/hp/Documents/inpe/dire_201912.txt', sep=',', header=None)

espe, dire = espe.values, dire.values

# energ = np.loadtxt('tmp/energ.txt')
# spread = np.loadtxt('tmp/spread.txt')

# espe = espe[:,::3]
# dire = dire[:,::3]
# energ = energ[:,::3]
# spread = spread[::3]

espe = espe * 5
local = 'Recife'
data = 'Dec/2012'
ws = None

#colocar 0.01 em vez nas direcoes igual a zero
dire[np.where(dire == 0)] = 0.01

ad = np.array(['Jan','Fev','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


fig = plt.figure(figsize=(8,9))
# fig = plt.figure()

ax = fig.add_subplot(111)

# a = 0
plt.plot(0,'w')

xmin, xmax, ymin, ymax = 0, 360, 0-9, 248+24
plt.axis([xmin, xmax, ymin, ymax])

ha = np.hanning(11)

# v6 = np.hanning(15)
# v61 = np.hanning(11)

# ad1 = np.array([31,29,31,30,31,30,31,31,30,31,30,31]) #???

# plt.show()
# stop

# plt.axis('off')
# plt.xticks(visible=False)
# plt.yticks(visible=False)
plt.xticks([])
plt.yticks([])

# plt.axis()

# arq2 = np.array(([ [1,0.00,0],
#                  [1,0.00,0],
#                  [1,0.55,0],
#                  [1,0.55,0],
#                  [1,1.00,0],
#                  [1,1.00,0],
#                  [0,1.00,0],
#                  [0,1.00,0] ]))

# col = np.array([0.7,0.7,0.7])

# left, bottom, width, height = 0, 0, 1, 1
# ax.set_position([left, bottom, width, height]) # set a new position

#########################################################################################
# titulo

pl.text(22,248+30,'DIRECTIONAL WAVE SPECTRUM EVOLUTION (DAAT/PLEDS) - %s - %s' \
                  %(local, data))#- Santos ' + date + ' - PNBOIA (AXYS 3M)')

#########################################################################################


# plt.show()
# stop


# #codigo de cores para faixas de periodos
# x = np.array([0.5,3,3,0.5]) + 0.5
# y = np.array([1,1,12,12]) - 3.5

#########################################################################################
# cria retangulos

# it =  #inicio dos triangulos
ee = 5 #espacamento para comecar a escrever 
cr = 65 #comprimento dos retangulos
at = 7 #altura dos triangulos

ax.add_patch(Rectangle((0*cr, -8), cr, at, alpha=0.8, facecolor='r'))
ax.text(ee+0*cr, -7, "21.3 - 16.0 s")

ax.add_patch(Rectangle((1*cr, -8), cr, at, alpha=0.8, facecolor='y'))
ax.text(ee+1*cr, -7, "12.8 - 07.1 s")

ax.add_patch(Rectangle((2*cr, -8), cr, at, alpha=0.8, facecolor='g'))
ax.text(ee+2*cr, -7, "06.4 - 04.0 s")

ax.add_patch(Rectangle((3*cr, -8), cr, at,alpha=0.8, facecolor='b'))
ax.text(ee+3*cr, -7, "03.7 - 01.5 s")

plt.draw()

#########################################################################################
# cria textos de legendas

# ax.add_patch(Rectangle((200, 0), cr, 6,alpha=0.8,facecolor='w'))
ax.text(ee+4*cr, -7, "5 div = 10 m/s & 0.1 m2", fontsize=9)

# ax.add_patch(Rectangle((250, 0), cr, 8,alpha=0.8,facecolor='w'))
# ax.text(ee+4*cr+50, 1, "5 div=0.1 m2")

# -------------------------------------------------
#linhas verticais
# divisoes a cada 20 graus - 18 setores

# 1    2     3    4    5    6    7    8
# N / NNE / NE / ENE / E / ESE / SE / SSE / S 

# y = np.array([20,283])
y = [10,248+24]
for i in range(0,360,20):
    x = [i,i]
    pl.plot(x,y,'k',alpha=0.2)

# -------------------------------------------------

#linhas horizontais (dias)
x = [0,360]
dia = 0
for i in range(10,248+10,8):
    dia += 1
    y = [i,i]
    pl.plot(x,y,'k',alpha=0.2)
    pl.text(-13,y[0]-2,str(dia).zfill(2))
    pl.text(363,y[0]-2,str(dia).zfill(2))


#linhas horizontais (3 horas)
x = [0, 360]
for i in range(10,248+10,2):
    y = [i,i]
    pl.plot(x,y,'k',alpha=0.1)



#########################################################################################
#plotagem do espectro direcional a cada dia por faixa de periodos
#plota de cima para baixo (de 31 para 1)

#eixo horizontal de direcao
# a = np.arange(310,720,20)
# a[pl.find(a>360)] = a[pl.find(a>360)] - 360

# coloca valores de direcao em graus
for i in range(0, 360, 20):
    pl.text(i+3, 2, str(i).zfill(3), color='red', fontsize=10)

# coloca valores de direcao rosa dos ventos
ld = np.array(['N','NE','E','SE','S','SW','W','NW'])

cont = 0
for i in range(0, 360, 45):
    pl.text(i+2, 10, ld[cont], color='k', weight='bold')
    cont += 1

# for i in range(len(ld)):

pl.text(1.15,80,'days in a month',rotation=90)
# pl.ylabel('days in a month')



# ld = np.array(['NW','N ','NE','E ','SE','S ','SW','W ','SW','W '])

# k = 1.55;
# d = np.arange(1.75,13,2.25)

# for i in range(len(ld)):
#     pl.text(k,20.8,ld[i],weight='bold')
#     k=k+2.25;
 


# stop

#w=[[1;6] [2;7] [3;8] [4;9]];
#bb = np.array([0.3]*10) #utilizado para montar o triangulo??

# yp = 20 #valor inicial do eixo y

# for t in range(dire.shape[1]-1,0,-3):
for t in range(dire.shape[1]-1,-1,-1):

    #define os vetores de direcao e espectro (energia)
    # s1 = dire[:,t]
    # s2 = espe[:,t]

    icor = -1 #indice das cores

    #varia as 4 faixas
    for f in [0,1,2,3]:

        # arq5 = arq2[i,:] #cor - fazer com 'r', 'b', ... ??

        icor += 1
        arq5 = ['r','y','g','b']

        # s11 = s1[i] #valor da direcao no tempo i
        # s12 = s2[i] #valor do espectro no tempo i

        # indice do eixo x das direcoes
        # xp = 1 + dire[i,t] * 0.0527
        # print yp

        # b1 = s11 / 20 #direcao (correcao para o eixo??)
        # b2 = s12 / 2  #espectro

        # if b1 > 0:

        #     b1 = b1 + 3 #ajuste da direcao

        #     if b1 > 18:
        #         b1 = b1 - 18

        #     b1 = b1 + 1 #o zero comeca em 111 (ou 11??)
        #     n1 = t + 9 + 10 #shift na escala vertical

        #     #monta um triangulo (x)
        #     v7 = np.linspace(b1-0.3 ,b1+0.3,len(v61))
        #     x = np.array(list(v7) + list(np.flipud(v7)))

        x = np.linspace(dire[f,t]-5, dire[f,t]+5, len(ha))

        # eixo x (direcao) com base na dp e spread
        # x = np.linspace(dire[f,t]-spread[t,-1]/2, dire[f,t]+spread[t,-1]/2, len(ha))
        y = ha * espe[f, t] + t

        pl.fill(x, y+10, arq5[icor], edgecolor='black', alpha=1)





        # pl.plot(xp,yp,'.',arq5[cor])
        # pl.plot(dire[f,t],t+10,'.',arq5[icor])
        

        # yp += 0.94



            #############################
            # teste para colocar o spread
            # vamos puxar os extremos do eixo y para +- spread/2

            # x[0:12] = x[0:12] - spread[t,-1]/2.0/20.0
            # x[12:] = x[12:] + spread[t,-1]/2.0/20.0

            # x = np.linspace(b1
            # y = n1 + b2

            # x = np.linspace(b1-1,b1+1,20)
            # y = np.hanning(len(x)) + n1*b2

            #############################
            
            

            #cria un hanning
            # y = np.array( list(n1+v61*b2) + list(n1*np.ones(len(v61))))
            # y = np.array( list(n1+v61*b2)) # + list(n1*np.ones(len(v61))))
            # y = np.hanning(len(x))*b2+n1
            # x =  

            # print y
            # x=np.arange(10) + b1 
            # y=hanning(len(x)) + n1*b2


            # x = x[:len(y)]

            # x[0] =  x[0]  - spread[t,-1]/2.0/20.0
            # x[-1] = x[-1] + spread[t,-1]/2.0/20.0


            # ??
            # z1 = 1
            # z2 = 1

            # if b2 * b1 > 0:

            # pl.fill(x,y,arq5[cor])
            # plt.plot(x, y, 'o')






#####################################################
#vento (u e v)

if ws != None:

    for t in np.arange(len(ws)-1,0,-1):

        s1 = ws[t]
        s2 = wd[t]

        if s1 > 0:

            s2 = s2/20
            st = t+9+10
            s2 = s2 + 3

            if s2 > 18:

                s2 = s2 - 18

            s2 = s2 + 1

            #s1 eh a velocidade do vento, colocada na escala certa
            #1 divisao=2m/s
            #s2 eha direcao em caixas de 18 graus
            #st eh a posicao de plotagem ao logo da vertical

            x = [s2-0.05,s2+0.05,s2+0.05,s2-0.05]
            y = [st,st,st+s1,st+s1]

            if s2 < 2:

                x = np.nan
                y = np.nan

            if s2 > 18:

                x = np.nan
                y = np.nan

            if s1 > 0:

                pl.fill(x,y,'w')

            if s1 > 10:

                pl.fill(x,y,'y')

            if s1 >20:

                pl.fill(x,y,'g')

plt.show()
# pl.savefig('fig/%s_%s.png' %(local, data) ,format='png')
pl.savefig('pleds_teste.png' ,format='png', bbox_inches='tight')
