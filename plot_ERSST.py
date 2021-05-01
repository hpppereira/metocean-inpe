# plot processed data om BESM database

import xray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pl

pathname = '/stornext/online2/ocean/DATABASE_FRE_VALIDITY/ERSST/'
filename = 'ERSST_SST_annualclim_1860_1890.nc'

d = xray.open_dataset(pathname + filename)

#plot figure
pl.imshow(d.sst.data)

pl.savefig('teste.png')


