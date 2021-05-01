# -*- coding: utf-8 -*-
#
# Script para gerar figuras das rodadas do BESM
# Data da ultima modificacao: 22/04/2017

import matplotlib.pylab as pl
import numpy as np
import os
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap as bm

plt.close('all')

def plot_field(X, lat, lon, vmin, vmax, step, pathname_fig, filename_fig, cmap=plt.get_cmap('jet'), ax=False, title=False, grid=False):

    m = bm(projection='cyl',
           llcrnrlat=lat.data.min(),
           urcrnrlat=lat.data.max(),
           llcrnrlon=lon.data.min(),
           urcrnrlon=lon.data.max(),
           lat_ts=0,
           resolution='c')
    
    lons, lats = np.meshgrid(lon, lat)

    if not ax: 
        f, ax = plt.subplots(figsize=(8, (X.shape[0] / float(X.shape[1])) * 8))
    m.ax = ax
    
    im = m.contourf(lons, lats, X, np.arange(vmin, vmax+step, step), \
                    latlon=True, cmap=cmap, extend='both', ax=ax)
    
    m.drawcoastlines()
    
    if grid: 
        m.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
        m.drawparallels(np.arange(-80, 80, 20), labels=[1,0,0,0])
    
    m.colorbar(im)
    
    if title: 
        ax.set_title(title)

    plt.savefig(pathname_fig + filename_fig)


def mean_longitude(X, name_x, var_x, lat_x, lon_x, time_x, depth_x,
                   Y, name_y, var_y, lat_y, lon_y, time_y, depth_y):
    
    xx = X.mean(time_x).mean(depth_x).mean(lon_x)[var_x]
    yy = Y.mean(time_y).mean(depth_y).mean(lon_y)[var_y]
    
    plt.plot(X[lat_x].data, xx.data, label=name_x)
    plt.plot(Y[lat_y].data, yy.data, label=name_y)
    plt.xlabel('Latitude')
    plt.ylabel(var_x)
    plt.axis('tight')
    pl.grid()
    pl.legend()




