# Organiza dados do ERSST para o
# banco de dados do FRE validity
# Henrique Pereira
# 2017/06/20
#
# Observacoes:
# Ainda nao esta finalizado. O Andre
# ja salvou os dados corretos na base de dados


# load libraries
import os
import numpy as np
import xray
from mpl_toolkits import basemap
from datetime import datetime

# read data
pathname = os.environ['HOME'] + '/Dropbox/inpe/data/ERSST/'
filename = 'sst.mnmean.v4.nc'

dd = xray.open_dataset(pathname + filename)

periods = np.array([['1860', '1890'],
	                ['1948', '1979'],
	                ['1979', '2015']])

# lat/lon of regular grid
lon = np.arange(0, 361)
lat = np.arange(-90, 91)

# interpolated regular grid
lons, lats = np.meshgrid(lon, lat)

for i in range(len(periods)):

	print i

	#slice choosed period
	a = dd.sel(time=slice(periods[i][0], periods[i][1]))

	#calculates a monthly climatology using the groupby machinery
	mclim = a.groupby('time.month').mean('time') #momthly mean
	aclim = a.mean('time') #annual mean

	#data with regular grid
	mgrid = np.zeros((len(mclim.sst), len(lat), len(lon)))

	#interp monthly clim
	for j in range(len(mclim.sst)):

		mgrid[i,:,:] = basemap.interp(datain=mclim.sst.data[j,:,:], xin=dd.lon.data, yin=dd.lat.data*-1, xout=lons, yout=lats)

    #interp annual clim
	agrid = basemap.interp(datain=aclim.sst.data, xin=dd.lon.data, yin=dd.lat.data*-1, xout=lons, yout=lats)

	# create netcdf of monthly clim
	d = {}
	d['time'] = ('time', mclim.month.data)
	d['lat'] = ('lat', lat)
	d['lon'] = ('lon', lon)
	d['sst'] = (['time', 'lat', 'lon'], mgrid)

	dset_from_dict = xray.Dataset(d)
	dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
	dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database/ersst_v4/'
	dset_from_dict.attrs['filename'] = 'sst.mnmean.v4.nc'
	dset_from_dict.attrs['time_interval'] = '%s - %s' %(periods[i][0], periods[i][1])
	dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
	dset_from_dict.attrs['temp_unit'] = 'Celsius'
	dset_from_dict.attrs['description'] = 'ERSST Monthly Climatology'

	dset_from_dict.to_netcdf('teste_monthly_%s.nc' %i)

	# create netcdf of annual clim
	d = {}
	d['lat'] = ('lat', lat)
	d['lon'] = ('lon', lon)
	d['sst'] = (['lat', 'lon'], agrid)

	dset_from_dict = xray.Dataset(d)
	dset_from_dict.attrs['creation_date'] = datetime.utcnow().strftime("%Y-%m-%d")
	dset_from_dict.attrs['dirname'] = '/stornext/online2/ocean/database/ersst_v4/'
	dset_from_dict.attrs['filename'] = 'sst.mnmean.v4.nc'
	dset_from_dict.attrs['time_interval'] = '%s - %s' %(periods[i][0], periods[i][1])
	dset_from_dict.attrs['latlon_unit'] = 'Dec. Degrees'
	dset_from_dict.attrs['temp_unit'] = 'Celsius'
	dset_from_dict.attrs['description'] = 'ERSST Annual Climatology'

	dset_from_dict.to_netcdf('teste_annual_%s.nc' %i)
