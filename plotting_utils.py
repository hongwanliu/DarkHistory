import sys
import numpy as np
import csv
from scipy.interpolate import interp1d
sys.path.append('../..')
import darkhistory.physics as phys

input_dir_default = '/Users/gregoryridgway/Desktop/Webplot_distortion/'
reion_strings = np.array(
    ['earliest', 'latest', 'FlexKnot_early', 'FlexKnot_late',
     'FlexKnot_early_1sig', 'FlexKnot_late_1sig', 'Tanh_early',
     'Tanh_late', 'Tanh_early_1sig', 'Tanh_late_1sig']
)


def plot_distortion(
    ax, run, run2=None,
    xlim=[4e-1, 1e6], ylim=[6e-30, 2e-26],
    label=None, color=None, title=None, alpha=1.0, leg=False,
    first=True, diff=False, rs_lim=None
):
    """ Plots the positive and negative distortion on a log-log plot

        Inputs
        ------
        ax : plot axis
        run : dict
            output of main.evolve()
        run2 : dict
            output of main.evolve() to be subtracted against run
        first : bool
            If True, label the positive and negative contributions
        diff : bool
            If True, plot the difference between the two distortions,
            run and run2

    """

    eng = run['distortion'].eng
    hplanck = phys.hbar * 2*np.pi
    nu = eng/hplanck
    convert = phys.nB * eng * hplanck * phys.c / (4*np.pi) * phys.ele * 1e4

    ax.loglog()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if rs_lim is not None:
        dist_copy = run['distortions'].copy()
        dist_copy.redshift(rs_lim)
        distortion = dist_copy.sum_specs(run['rs'] >= rs_lim)
    else:
        distortion = run['distortion']

    if first:
        ax.plot(nu/1e9, np.ones_like(nu), color='k',
                linestyle='solid', label='Positive Distortion')
        ax.plot(nu/1e9, np.ones_like(nu), color='k',
                linestyle='--', label='Negative Distortion')

    if not diff:
        ax.plot(nu/1e9, convert * distortion.dNdE,
                color=color, linewidth=1.4,
                label=label, linestyle='solid', alpha=alpha)

        ax.plot(nu/1e9, -convert * distortion.dNdE,
                color=ax.get_lines()[-1].get_color(), linewidth=1.4,
                linestyle='--', alpha=alpha)

    else:
        ax.plot(nu/1e9, convert * (distortion.dNdE - run2['distortion'].dNdE),
                color=color, linewidth=1.4,
                label=label, linestyle='solid', alpha=alpha)

        ax.plot(nu/1e9, -convert * (distortion.dNdE - run2['distortion'].dNdE),
                color=ax.get_lines()[-1].get_color(), linewidth=1.4,
                linestyle='--', alpha=alpha)

#     ax.set_xticks([1e1,1e2,1e3,1e4,1e5,1e6])

    ax.set_xlabel(r'Frequency, $\nu$ [GHz]', fontsize=20)
    ax.set_ylabel(r'Intensity, $I_{\nu}$ ' +
                  '[J s$^{-1}$ m$^{-2}$ Hz$^{-1}$ sr$^{-1}$]', fontsize=20)
    ax.set_title(title, fontsize=16)

    if leg:
        ax.legend()
        # leg = axarr[0].legend(fontsize=12, loc=1)
        # leg.set_title(r'$m_\chi = 200 MeV$, $\tau = 2 \times 10^{25}s$')
        # axarr[0].setp(leg.get_title(),fontsize='12')


def download_plot(file, input_dir=input_dir_default):
    with open(input_dir+file+'.csv') as csvfile:
        reader = csv.reader(csvfile)
        goods = []
        for row in reader:
            goods.append([float(r) for r in row])

        goods = np.array(goods)

    return goods


def make_reion_interp_func(string, He_bump=False):
    input_dir = '/Users/gregoryridgway/Dropbox (MIT)/' +\
                'Late_Time_Energy_Injection/'
    Planck_data = []
    with open(input_dir+'/reion_models/Planck_' +
              string+'.csv') as csvfile:
        reader = csv.reader(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            Planck_data.append([float(row[0]), float(row[1])])
    Planck_data = np.array(Planck_data)

    # fix normalization
    if string == 'FlexKnot_early':
        norm_fac = Planck_data[-2, 1]
    else:
        norm_fac = Planck_data[0, 1]

        # I WebPlot Digitized poorly, so I re-zero
        if string == 'FlexKnot_late':
            Planck_data[26:, 1] = 0
        elif string == 'Tanh_late':
            Planck_data[63:, 1] = 0

    Planck_data[:, 1] = (1+2*phys.chi)*Planck_data[:, 1]/norm_fac

    # convert from z to rs
    Planck_data[:, 0] = 1+Planck_data[:, 0]

    fac = 2
    if not He_bump:
        Planck_data[Planck_data[:, 1] > 1+phys.chi, 1] = 1+phys.chi
        fac = 1

    return interp1d(Planck_data[:, 0], Planck_data[:, 1],
                    bounds_error=False, fill_value=(1+fac*phys.chi, 0),
                    kind='linear')

# Make interpolation functions for each Planck2018 reionization history
# reion_interps = {string : make_reion_interp_func(string) for string in reion_strings}
# bump_interp = make_reion_interp_func('Tanh_late', True)
