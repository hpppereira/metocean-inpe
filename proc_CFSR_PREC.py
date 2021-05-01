# Organiza dados do CFSR para o
# banco de dados do FRE validity
# Henrique Pereira
# 2017/06/26
# O que o script faz:
# 1) junta os arquivos nc
# 2) Calcula climatologias (mensal e anual)
# 3) Faz o regradeamento
# 4) Salva nc no formato padrao do banco

# load libraries
import os
import numpy as np
import xray
from mpl_toolkits import basemap
from datetime import datetime

# pathname
pathname = '../data/CFSR/prec/'

# diretorios com variaveis
#dirvars = ['cld','prec','sphum','temp']
#dv = 'prec'

# loop para variar os arquivos de cada diretorio
#for dv in dirvars[:1]:

#    print dv

    #lista arquivos dentro de cada diretorio
list_nc = np.sort(os.listdir(pathname))

# contador
cont = 0

# loop para juntar os arquivos (varia os arquivos nc)
for arq in list_nc:

    print arq
    cont += 1

    # carrega o primeiro arquivo que sera concatenado
    if cont == 1:
        dd_concat = xray.open_dataset(pathname + '/' + arq)
        #dd_concat = dd_concat[:,0,:,:]

        #print dd_concat

    else:

        dd_aux = xray.open_dataset(pathname + '/' + arq)

        #print dd_aux

        dd_concat = xray.concat([dd_concat, dd_aux], dim='time')

stop

#faz a climatologia
# mclim = dd_concat.groupby('time.month').mean('time') #monthly clim
# aclim = dd_concat.mean('time') #annual clim

# lat/lon of regular grid
lon = np.arange(-180, 181)
lat = np.arange(-90, 91)

# regular grid
lons, lats = np.meshgrid(lon, lat)

# #regrid monthly clim

#data with regular grid
rgrid = np.zeros((len(dd_concat), len(lat), len(lon)))

for j in range(len(rclim)):
    rgrid[j,:,:] = basemap.interp(datain=dd_concat.data[j,:,:],
                                  xin=dd_concat.longitude.data,
                                  yin=dd_concat.latitude.data*-1,
                                  xout=lons,
                                  yout=lats)

# #regrid annual clim
# agrid = basemap.interp(datain=aclim.data, xin=aclim.longitude.data, yin=aclim.latitude.data*-1, xout=lons, yout=lats)

# stop

# #create netcdf of monthly clim
# d = {}
# d['time'] = ('time', mclim.month.data)
# d['latitude'] = ('latitude', np.arange(-90,91)*-1)
# d['longitude'] = ('longitude', np.arange(0,361))
# d['prec'] = (['time', 'latitude', 'longitude'], mgrid)

# dset_from_dict = xray.Dataset(d)
# dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
# dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database_new/reanalysis/NOAA/CFSR/day/global/'
# dset_from_dict.attrs['time_interval'] = '1981 - 2010'
# dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
# dset_from_dict.attrs['description'] = 'CFSR Total Precipitation Monthly Climatology'

# dset_from_dict.longitude.attrs['units'] = 'longitude'
# dset_from_dict.latitude.attrs['units'] = 'latitude'
# dset_from_dict.prec.attrs['units'] = 'mm/day'

# dset_from_dict.to_netcdf('CFSR_PREC_monthclim_1981_2010.nc')


# #create netcdf of annual clim
# d = {}
# d['latitude'] = ('latitude', np.arange(-90,91))
# d['longitude'] = ('longitude', np.arange(0,361))
# d['prec'] = (['latitude', 'longitude'], agrid)

# dset_from_dict = xray.Dataset(d)
# dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
# dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database_new/reanalysis/NOAA/CFSR/day/global/'
# dset_from_dict.attrs['time_interval'] = '1981 - 2010'
# dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
# dset_from_dict.attrs['description'] = 'CFSR Total Precipitation Annual Climatology'

# dset_from_dict.longitude.attrs['units'] = 'longitude'
# dset_from_dict.latitude.attrs['units'] = 'latitude'
# dset_from_dict.prec.attrs['units'] = 'mm/day'

# dset_from_dict.to_netcdf('CFSR_PREC_annualclim_1981_2010.nc')

