# Processamento dos dados da boia Spotter
# do INPE/LOA na Antártica

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.signal import find_peaks
from pandas.plotting import register_matplotlib_converters
import pycwt as wavelet
from pycwt.helpers import find

register_matplotlib_converters()

plt.close('all')

def matlab2datetime(matlab_datenum):
    """
    Cria datetime
    """
    day = datetime.fromordinal(int(matlab_datenum))
    dayfrac = timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    return day + dayfrac

def plot_hstpdp(df):
    """
    Plot Hs, Tp e Dp
    """
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(311)
    ax1.plot(df.Hs)
    ax1.set_ylabel('Hs (m)')
    ax1.grid()
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(df.Tp)
    ax2.set_ylabel('Tp (s)')
    ax2.grid()
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(df.Dir_p, label='Dir_p')
    ax3.set_ylabel('Dp (º)')
    ax3.grid()
    return fig

def plot_espectro_1d(t, f, s, d, hs, tp, dp, ifaixas):
    """
    Plot do espectro 1D
    t - datetime do espectro
    s - vetor do espectro 1D
    d - vetor de direcao principal por frequencia
    """
    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_subplot(211)
    ax1.set_title(t + '\nHs={:.1f} m, Tp={:.1f} s, Dp={:.1f}º'.format(hs, tp, dp))
    ax1.plot(f, s, 'b')
    ax1.fill_between(f[ifaixas['ifx1']], 0, max(s), color='k', alpha=0.2)
    ax1.fill_between(f[ifaixas['ifx2']], 0, max(s), color='r', alpha=0.2)
    ax1.fill_between(f[ifaixas['ifx3']], 0, max(s), color='g', alpha=0.2)
    ax1.fill_between(f[ifaixas['ifx4']], 0, max(s), color='y', alpha=0.2)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, s.max())
    ax1.grid()
    ax1.set_ylabel('Energia (m²/Hz)')
    # ax1.set_ylim(0, 20)
    ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.set_title(t + '\nHs={:.1f} m, Tp={:.1f} s, Dp={:.1f}º'.format(hs, tp, dp))
    ax2.plot(f, d, 'b')
    ax2.fill_between(f[ifaixas['ifx1']], 0, 360, color='k', alpha=0.2)
    ax2.fill_between(f[ifaixas['ifx2']], 0, 360, color='r', alpha=0.2)
    ax2.fill_between(f[ifaixas['ifx3']], 0, 360, color='g', alpha=0.2)
    ax2.fill_between(f[ifaixas['ifx4']], 0, 360, color='y', alpha=0.2)
    ax2.set_xlim(0, 0.5)
    ax2.grid()
    ax2.set_yticks(np.arange(0, 361, 45))
    ax2.set_ylabel('Direção (º)')
    ax2.set_xlabel('Frequência (Hz)')
    return fig

def plot_evolspec(date, f, sp, tipo):
    """
    Plot evolução espectral com função contour
    """
    fig=plt.figure(figsize=(10,6),facecolor='w')
    fig.subplots_adjust(hspace=0.25)
    fig.tight_layout()
    # ttl = ax1.title
    # ttl.set_position([.5, 1.05])
    # lvls = np.arange(0, sp.max().max(), .5)
    lvls = np.arange(0, 12.5, .5)
    if tipo == 'contour':
        ax1 = fig.add_subplot(111)
        CF = ax1.contourf(date, f, sp,
                          locator=None,
                          levels=lvls,
                          cmap=plt.cm.jet)

        cbar = plt.colorbar(CF,
                            # ticks=np.arange(0,1.1,0.1),
                            format='%.1f', label='m²/Hz',
                            orientation='horizontal', fraction=0.1, pad=0.15)
        ax1.set_ylabel('Frequência (Hz)')
        ax1.grid('on')
        ax1.set_ylim(0, 0.3)
        plt.xticks(rotation=0)
    elif tipo == 'surface':
        ax1 = fig.add_subplot(111, projection='3d')
        x = np.arange(0, len(sp.T))
        X, Y = np.meshgrid(x, f[:40], indexing='ij')
        surf = ax1.plot_surface(X, Y, sp.T[:,:40], cmap=cm.jet,
                               linewidth=0, antialiased=False)
        # ax1.set_ylim(0, 0.4)
        ax1.set_xlabel('Horas')
        ax1.set_ylabel('Frequência (Hz)')
        ax1.set_zlabel('Energia (m²/Hz)')
        # ax1.xaxis.set_ticks(x[::1000])
        # ax1.xaxis.set_ticklabels(df.index[::1000])
        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax1.zaxis.set_major_locator(LinearLocator(10))
        # ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax1.view_init(elev=30., azim=135)
        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def plot_espe_dire(espe, dire):
    """
    plotagem da energia e direção das faixas
    """
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(211)
    ax1.plot(espe[['f1', 'f2', 'f3', 'f4']])
    ax1.set_ylabel('Energia (m²/Hz)')
    ax1.legend(['faixa 1', 'faixa 2', 'faixa 3', 'faixa 4'], ncol=2)
    ax1.grid()
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(dire[['f1', 'f2', 'f3', 'f4']])
    ax2.set_yticks(np.arange(0, 361, 45))
    ax2.set_ylabel('Direção (º)')
    ax2.grid()
    return fig

def plot_wavelet(dat, dt):
    """
    """
    # Remove os NaN
    # dat = dat.interpolate().dropna()

    title = 'Altura Significativa'
    label = 'Wavelets'
    units = 'metros'
    t0 = 0 #1871.0

    # We write the following code to detrend and normalize the input data by its
    # standard deviation. Sometimes detrending is not necessary and simply
    # removing the mean value is good enough. However, if your dataset has a well
    # defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available in the
    # above mentioned website, it is strongly advised to perform detrending.
    # Here, we fit a one-degree polynomial function and then subtract it from the
    # original data.

    #decomp = seasonal_decompose(dat, model="additive", freq=8760)
    dat_noseason = dat #- decomp.seasonal
    #dat_notrend = dat_noseason - decomp.trend       
    #dat_notrend = dat_notrend.values
    times = dat.index        
    t = times.to_julian_date()  #np.arange(0, N) * dt + t0
    dat = dat_noseason.values 
    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset

    # We also create a time array in years.
    N = dat.size

    #data_norm = (data - data.min()) / (data.max() - data.min())        
    #data_detrended = signal.detrend(data_norm)
    #data = data_detrended
    #variance = np.std(data)**2
    #mean=np.mean(data)
    #data = (data - np.mean(data))/np.sqrt(variance)
    #print("mean=",mean)
    #print("std=", np.sqrt(variance))


    # The next step is to define some parameters of our wavelet analysis. We
    # select the mother wavelet, in this case the Morlet wavelet with
    # :math:`\omega_0=6`.
    mother = wavelet.Morlet(6)
    s0 = 8 * dt   # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12   # Twelve sub-octaves per octaves
    J = 8 / dj    # Seven powers of two with dj sub-octaves
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # The following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized our
    # input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                    mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # We calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # We could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is significant
    # where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                            significance_level=0.95,
                                            wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Then, we calculate the global wavelet spectrum and determine its
    # significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)


    # We also calculate the scale average between 2 years and 8 years, and its
    # significance level.        
    sel = find((period >= 0.5) & (period < 5))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                            significance_level=0.95,
                                            dof=[scales[sel[0]],
                                                    scales[sel[-1]]],
                                            wavelet=mother)

    # Finally, we plot our results in four different subplots containing the
    # (i) original series anomaly and the inverse wavelet transform; (ii) the
    # wavelet power spectrum (iii) the global wavelet and Fourier spectra ; and
    # (iv) the range averaged wavelet spectrum. In all sub-plots the significance
    # levels are either included as dotted lines or as filled contour lines.

    # Prepare the figure
    plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(**figprops)

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    #ax.plot(times, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(times, dat, 'k', linewidth=1.5)
    ax.set_title('a) {}'.format(title))
    ax.set_ylabel(r'[{}]'.format(units))

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    #bx = plt.axes([0.1, 0.1, 0.65, 0.58], sharex=ax)
    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8,16, 32, 64, 128]
    bx.contourf(times, np.log2(period), np.log2(power), np.log2(levels),
            extend='both', cmap=plt.cm.RdBu_r)
    extent = [times.min(), times.max(), 0, max(period)]
    bx.contour(times, np.log2(period), sig95, [-99, 1], colors='k', linewidths=1,
            extent=extent)
    bx.fill(np.concatenate([times, times[-1:] + timedelta(hours=1), times[-1:] + timedelta(hours=1),
                            times[:1] - timedelta(hours=1), times[:1] - timedelta(hours=1)]),
            np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                            np.log2(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    bx.set_title('b) Espectro de potência das wavelets ({})'.format(mother.name))
    bx.set_ylabel('Período (dias)')
    #
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)

    #bx.invert_yaxis()
    #ax = plt.gca()
    #bx.set_ylim(bx.get_ylim()[::-1])

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    #cx = plt.axes([0.77, 0.1, 0.2, 0.58], sharey=bx)
    cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, np.log2(period), 'k--',zorder=2)
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc',zorder=2)
    cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc',
            linewidth=1.,zorder=1)
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Espectro Global da Wavelet')
    cx.set_xlabel(r'Potência [({})^2]'.format(units))
    cx.set_xlim([0, glbl_power.max() + 50*var])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)
    cx.invert_yaxis()
    plt.setp(cx.get_yticklabels(), visible=False)

    # Fourth sub-plot, the scale averaged wavelet spectrum.
    dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=bx)
    dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    dx.plot(times, scale_avg, 'k-', linewidth=1.5)
    dx.set_title('d) Potência média para escala de {}-{} dias'.format(0.5, 5))
    dx.set_xlabel('Data')
    dx.set_ylabel(r'Variância média [{}]'.format(units))
    dx.set_xlim([times.min(), times.max()])                       
    # logo = plt.imread('logo_atmosmarine_10kb.jpg')
    # ax.figure.figimage(logo, 620, 10, alpha=1, zorder=10)
    # plt.close('all')
    return fig



if __name__ == "__main__":

    pathname = '/home/hp/Documents/inpe/'

    df = pd.read_csv(pathname + 'dados/spotter_antartica/Wave_Data_filtered2.csv') #, encoding='ISO-8859-1')

    # data em datetime
    df['date'] = [matlab2datetime(df.time[i]) for i in range(0, len(df))]   #para datetime do python

    # coloca data como index
    df.set_index('date', inplace=True)

    # vetor de frequencia
    f = np.linspace(0, 1.25, 129)

    # matriz com espectro 1d, e coeficientes a1, b1, a2 e b2
    sp = df.iloc[:,11:11+129]#.values
    a1 = df.iloc[:,11+1*129:11+2*129].values
    b1 = df.iloc[:,11+2*129:11+3*129].values
    a2 = df.iloc[:,11+3*129:11+4*129].values
    b2 = df.iloc[:,11+4*129:11+5*129].values

    # indices das faixas
    ifaixas = {}
    ifaixas['ifx1'] = np.arange(4, 9)   # 25.60 a 12.8
    ifaixas['ifx2'] = np.arange(9, 16) # 11.37 a 06.82  
    ifaixas['ifx3'] = np.arange(16, 25) # 06.40 a 04.26
    ifaixas['ifx4'] = np.arange(25, 50) # 04.09 a 02.08 

    # calcula direcao de onda
    dirp = []
    # cria variaveis espe e dire para 4 faixas (PLEDS)
    espe = pd.DataFrame(np.zeros((df.shape[0], 4)), index=df.index,
                        columns=['f1', 'f2', 'f3', 'f4'])
    dire = espe.copy()
    for i in range(len(df))[:1]:
        print ('Calculando espe e dire - {} de {}'.format(i, len(df)))
        d = np.array([np.angle(np.complex(b1[i,j],a1[i,j]), deg=True) for j in range(a1.shape[1])])
        d = d + 180
        d[np.where(d < 0)] = d[np.where(d < 0)] + 360
        d[np.where(d > 360)] = d[np.where(d > 360)] - 360
        dirp.append(d)

        espe['f1'][i] = sp.iloc[i,ifaixas['ifx1']].sum()
        espe['f2'][i] = sp.iloc[i,ifaixas['ifx2']].sum()
        espe['f3'][i] = sp.iloc[i,ifaixas['ifx3']].sum()
        espe['f4'][i] = sp.iloc[i,ifaixas['ifx4']].sum()

        dire['f1'][i] = d[ifaixas['ifx1']].mean()
        dire['f2'][i] = d[ifaixas['ifx2']].mean()
        dire['f3'][i] = d[ifaixas['ifx3']].mean()
        dire['f4'][i] = d[ifaixas['ifx4']].mean()

    dirp = np.array(dirp)

    # plotagem dos espectros 1d e direcao media
    for i in range(0, len(df), 1)[:1]:
        print ('Plotando espectros 1D - {} de {}'.format(i, len(df)))
        t = df.index[i].strftime('%Y-%m-%d %H:%M')
        s = sp.iloc[i,:].values
        d = dirp[i,:]
        fig = plot_espectro_1d(t, f, s, d, hs=df.Hs[i], tp=df.Tp[i], dp=df.Dir_p[i], ifaixas=ifaixas)
        fig.savefig(pathname + 'figs/espec1d/espec1d_{}.png'.format(df.index[i]))
        plt.close('all')

    # reamostra espe e dire em hora
    espe = espe.resample('3H').mean()
    dire = dire.resample('3H').mean()

    # fig = plot_espe_dire(espe, dire)
    # fig.savefig(pathname + 'figs/espe_dire_pleds.png', bbox_inches='tight')

    # fig = plot_hstpdp(df)
    # fig.savefig(pathname + 'figs/hstpdp.png', bbox_inches='tight')

    # fig = plot_evolspec(df.index, f, sp.T, tipo='contour')
    # fig.savefig(pathname + 'figs/evolspec_contour.png', bbox_inches='tight')

    fig = plot_wavelet(dat=df.Hs, dt=30.0/60.0/24)
    fig.savefig(pathname + 'figs/wavelet_hs.png', DPI=100)

    # fig = plot_evolspec(df.index, f, sp.T, tipo='surface')
    # fig.savefig(pathname + 'figs/evolspec_surface.png')

    espe1 = pd.DataFrame(index=pd.date_range('2019-11-01 00:00','2020-03-31 23:00'), columns=['f1', 'f1', 'f2', 'f4']).resample('3H').asfreq()
    espe1['2019-11'] = espe['2019-11']
    espe1['2019-12'] = espe['2019-12']
    espe1['2020-01'] = espe['2020-01']
    espe1['2020-02'] = espe['2020-02']
    espe1['2020-03'] = espe['2020-03']
    espe1[espe1.isna()] = 0.001

    dire1 = pd.DataFrame(index=pd.date_range('2019-11-01 00:00','2020-03-31 23:00'), columns=['f1', 'f1', 'f2', 'f4']).resample('3H').asfreq()
    dire1['2019-11'] = dire['2019-11']
    dire1['2019-12'] = dire['2019-12']
    dire1['2020-01'] = dire['2020-01']
    dire1['2020-02'] = dire['2020-02']
    dire1['2020-03'] = dire['2020-03']
    dire1[dire1.isna()] = 0.001

    np.savetxt(pathname + 'espe_201911.txt', espe1['2019-11'].T.values, fmt='%.2f', delimiter=',')
    np.savetxt(pathname + 'dire_201911.txt', dire1['2019-11'].T.values, fmt='%.2f', delimiter=',')

    np.savetxt(pathname + 'espe_201912.txt', espe1['2019-12'].T.values, fmt='%.2f', delimiter=',')
    np.savetxt(pathname + 'dire_201912.txt', dire1['2019-12'].T.values, fmt='%.2f', delimiter=',')

    np.savetxt(pathname + 'espe_202001.txt', espe1['2020-01'].T.values, fmt='%.2f', delimiter=',')
    np.savetxt(pathname + 'dire_202001.txt', dire1['2020-01'].T.values, fmt='%.2f', delimiter=',')

    np.savetxt(pathname + 'espe_202002.txt', espe1['2020-02'].T.values, fmt='%.2f', delimiter=',')
    np.savetxt(pathname + 'dire_202002.txt', dire1['2020-02'].T.values, fmt='%.2f', delimiter=',')

    np.savetxt(pathname + 'espe_202003.txt', espe1['2020-03'].T.values, fmt='%.2f', delimiter=',')
    np.savetxt(pathname + 'dire_202003.txt', dire1['2020-03'].T.values, fmt='%.2f', delimiter=',')

    # os.system('matlab -nodisplay -nodesktop -r "run pleds_spotter_antartica.m"')
    # os.system('''matlab -nodisplay -nosplash -nodesktop -r "run('pleds_spotter_antartica.m');exit;"''')
    # os.system('eog pleds_spotter_antartica201912.png')

    # # cria video com espectro 1d
    # cria_video = "ffmpeg -framerate 3 -pattern_type glob -i '{}*.png' -c:v libx264 -pix_fmt yuv420p {}out.mp4".format(pathname + 'figs/espec1d/', pathname + 'figs/espec1d/')
    # os.system('rm {}out.mp4'.format(pathname + 'figs/espec1d/'))
    # os.system(cria_video)

    plt.show()
