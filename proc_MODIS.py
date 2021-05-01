# Organiza dados do LEVITUS para o
# banco de dados do FRE validity
# Henrique Pereira
# 2017/07/05
# O que o script faz:
# 1) Calcula climatologias para TEMP e SALT (mensal e anual)
# 2) Salva nc no formato padrao do banco

# load libraries
import os
import numpy as np
import xray
from mpl_toolkits import basemap
from datetime import datetime

# pathname
pathname = '/stornext/online2/ocean/database/'
filename = 'levitus_ewg.nc'

ds = xray.open_dataset(pathname + filename)

# faz a climatologia
mclim = ds.groupby('TIME.month').mean('TIME') #monthly clim
aclim = ds.mean('TIME') #annual clim

# lat/lon of regular grid
lon = np.arange(-180, 181)
lon = np.arange(0, 361)
lat = np.arange(-90, 91)

# interpolated regular grid
lons, lats = np.meshgrid(lon, lat)

#regrid monthly clim

#data with regular grid (monthly climatology)
mgrid_temp = np.zeros((len(mclim.temp), len(mclim.DEPTH), len(lat), len(lon)))
mgrid_salt = np.zeros((len(mclim.salt), len(mclim.DEPTH), len(lat), len(lon)))

# temperature
for t in range(len(mgrid_temp)): #varia o tempo
    for j in range(mclim.temp.data.shape[1]): #varia as profundidades
        mgrid_temp[t,j,:,:] = basemap.interp(datain=mclim.temp.data[t,j,:,:], xin=mclim.LONGITUDE.data, yin=mclim.LATITUDE.data, xout=lons, yout=lats)
        mgrid_salt[t,j,:,:] = basemap.interp(datain=mclim.salt.data[t,j,:,:], xin=mclim.LONGITUDE.data, yin=mclim.LATITUDE.data, xout=lons, yout=lats)

agrid_temp = np.zeros((len(mclim.DEPTH), len(lat), len(lon)))
agrid_salt = np.zeros((len(mclim.DEPTH), len(lat), len(lon)))

#regrid (annual clim)
for j in range(aclim.temp.data.shape[0]): #varia as profundidades
    agrid_temp[j,:,:] = basemap.interp(datain=aclim.temp.data[j,:,:], xin=aclim.LONGITUDE.data, yin=aclim.LATITUDE.data, xout=lons, yout=lats)
    agrid_salt[j,:,:] = basemap.interp(datain=aclim.salt.data[j,:,:], xin=aclim.LONGITUDE.data, yin=aclim.LATITUDE.data, xout=lons, yout=lats)


#create netcdf of monthly clim

for var in ['temp', 'salt']:

    if var == 'temp':
        mgrid = mgrid_temp
        agrid = agrid_temp
        descm = 'LEVITUS Ocean Temperature Monthly Climatology'
        desca = 'LEVITUS Ocean Temperature Annual Climatology'
        unit = 'Celsius Degrees'
        filenamem = 'LEVITUS_TEMP_monthclim_1935_1978.nc'
        filenamea = 'LEVITUS_TEMP_annualclim_1935_1978.nc'
    elif var == 'salt':
        mgrid = mgrid_salt
        agrid = agrid_salt
        descm = 'LEVITUS Ocean Salinity Monthly Climatology'
        desca = 'LEVITUS Ocean Salinity Annual Climatology'
        unit = 'PSU'
        filenamem = 'LEVITUS_SALT_monthclim_1935_1978.nc'
        filenamea = 'LEVITUS_SALT_annualclim_1935_1978.nc'

    d = {}
    d['time'] = ('time', mclim[var].month.data)
    d['depth'] = ('depth', mclim[var].DEPTH.data)
    d['latitude'] = ('latitude', np.arange(-90,91)) #*-1)
    d['longitude'] = ('longitude', np.arange(0,361))
    d[var] = (['time', 'depth', 'latitude', 'longitude'], mgrid)

    dset_from_dict = xray.Dataset(d)
    dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
    dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database/'
    dset_from_dict.attrs['time_interval'] = '1935 - 1978'
    dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
    dset_from_dict.attrs['description'] = descm

    dset_from_dict.depth.attrs['units'] = 'Meters'
    dset_from_dict.longitude.attrs['units'] = 'longitude'
    dset_from_dict.latitude.attrs['units'] = 'latitude'
    dset_from_dict[var].attrs['units'] = unit

    dset_from_dict.to_netcdf(filenamem)

    #create netcdf of annual clim
    d = {}
    d['depth'] = ('depth', mclim[var].DEPTH.data)
    d['latitude'] = ('latitude', np.arange(-90,91))
    d['longitude'] = ('longitude', np.arange(0,361))
    d[var] = (['depth', 'latitude', 'longitude'], agrid)

    dset_from_dict = xray.Dataset(d)
    dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
    dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database/'
    dset_from_dict.attrs['time_interval'] = '1935 - 1978'
    dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
    dset_from_dict.attrs['description'] = desca

    dset_from_dict.depth.attrs['units'] = 'Meters'
    dset_from_dict.longitude.attrs['units'] = 'longitude'
    dset_from_dict.latitude.attrs['units'] = 'latitude'
    dset_from_dict[var].attrs['units'] = unit

    dset_from_dict.to_netcdf(filenamea)

