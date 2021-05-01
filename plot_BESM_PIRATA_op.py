# Compara dados de CTD e XBT do projeto
# PIRATA com o modelo BESM
# Inputs:
# pathname - pathname of CTD data


#import libraries
import os
import xray
import numpy as np

def CTD(pathname):

    listpira = []
    for i in np.sort(os.listdir(pathname)):
        if i.endswith('.nc'):
            listpira.append(i)

    pira = xray.open_dataset(pathname + listpira[0])

    infos = {'year ': pira.start_time[:4],
             'month': pira.start_time[5:7],
             'day'  : pira.start_time[8:10],
             'hour' : pira.start_time[11:13],
             'lat'  : pira.lat,
             'lon'  : pira.lon}

    return pira, infos

pira, infos = CTD(pathname='/stornext/home/henrique.prado/project/obsdata/pirata_ctd/')

print pira
print infos
