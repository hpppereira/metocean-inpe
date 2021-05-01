# Converte dados do summary para netcdf

# Date    Year    Julian Date Zero Crossings  Ave. Ht.    
# Ave. Per.   Max Ht. Sig. Wave    Sig. Per.  Peak Per.(Tp)
#Peak Per.(READ) HM0 Mean Theta  Sigma Theta H1/10   T.H1/10 Mean Per.(Tz)


import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pathname = os.environ['HOME'] + '/gdrive/pnboia/dados/santos/'
    filename = 'san_summary.txt'

    df = pd.read_table(pathname + filename, delimiter='\t+', header=0,
    names = ['Date', 'Year', 'Julian_Date', 'Zero_Crossings', 'Ave_Ht', 'Ave_Per',
             'Max_Ht', 'Sig_Wave', 'Sig_Per', 'Peak_Per', 'Peak_Per_READ', 'HM0',
             'Mean_Theta', 'Sigma_Theta', 'H1_10', 'T_H1_10', 'Mean_Per'],
             index_col='Date', parse_dates=True, engine='python')

    df.sort_index(inplace=True)

    ds = df.to_xarray()

    ds.to_netcdf('pnboia_santos.nc', format='NETCDF4')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.HM0)
    ax1.set_ylabel('Hm0 (m)')
    ax1.set_title('PNBOIA - Santos')

    plt.show()
