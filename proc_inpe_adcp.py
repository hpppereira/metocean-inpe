# Processamento dos dados do ADCP 
# do Vital de Oliveira - INPE

import os
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import geoplot
import imageio

plt.close('all')

def read_adcp_mat(pathname, filename):
    """
    """
    mat = scipy.io.loadmat(pathname + filename)

    tday = mat['adp']['tday'][0][0][0]
    depth = mat['adp']['depth'][0][0][:,0]
    lon = mat['adp']['lon'][0][0][0]
    lat = mat['adp']['lat'][0][0][0]
    u = mat['adp']['u'][0][0]
    v = mat['adp']['v'][0][0]

    return tday, depth, lon, lat, u, v

def plot_track_shp(lcosta, ilhas, lat, lon, u, v, i):
    """
    plotagem do mapa com track
    """

    print (i)

    fig = plt.figure(figsize=(8,8))

    ax1 = fig.add_subplot(111)
    geoplot.polyplot(ilhas, ax=ax1, color='k', fill=False)
    lcosta.plot(ax=ax1, color='k')
    ax1.set_xlim(-39.57, -35.64)
    ax1.set_ylim(-15.67, -11.27)
    ax1.plot(lon, lat)
    ax1.plot(lon[i], lat[i], 'ro', markersize=6)

    qwind = ax1.quiver(lon[i], lat[i], u[0,i], v[0,i],
                       scale=10, pivot='tail', width=0.0030)#, headwidth=2)
    ax1.grid()
    ax1.set(xlabel='Longitude (º)', ylabel='Latitude (º)',
           title=None)
    ax1.set_aspect('equal', 'box')

    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close('all')
    return image

if __name__ == "__main__":

    pathname = os.environ['HOME'] + '/gdrive/Comissao_vital_de_oliveira/'
    path_shp = os.environ['HOME'] + '/gdrive/svarqueiro/dados/shapefiles/'
    filename = 'Dados_ADCP_LACN_75KHz.mat'
    file_lcosta_shp = 'LINHA_DE_COSTA_IMAGEM_GEOCOVER_SIRGAS_2000.shp'
    file_ilhas_shp = 'ILHAS_IMAGEM_GEOCOVER_SIRGAS_2000.shp'

    tday, depth, lon, lat, u, v = read_adcp_mat(pathname, filename)

    # leitura dos arquivos de linha de costa e ilhas do rio
    lcosta = gpd.read_file(path_shp + file_lcosta_shp)
    ilhas = gpd.read_file(path_shp + file_ilhas_shp)

    # for i in np.arange(0,5,1):
    #     plot_track_shp(lcosta, ilhas, lat, lon, u, v, i)
    #     plt.show()

    imageio.mimsave('./track_adcp_vital.mp4', [plot_track_shp(lcosta, ilhas, lat, lon, u, v, i) 
                                         for i in np.arange(0,len(tday),10)], fps=2)

    plt.show()


