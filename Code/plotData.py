import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# constants
dataPath = '../Data/'
outputPath = '../Plots/trends/'

filenames = [
                'scan_585_z1_04_data.txt',
                'scan_585_z1_05_data.txt',
                'scan_585_z1_06_data.txt',
                'scan_585_z1_07_data.txt',
                'scan_585_z1_08_data.txt',
                'scan_585_z1_09_data.txt',
                'scan_585_z1_10_data.txt',
                'scan_640_z1_01_data.txt',
                'scan_640_z1_02_data.txt'
            ]

# read data from txt file
def readData(fname):

    return np.loadtxt(dataPath + fname, delimiter=';')


def lin(x, a, b):
    return a*x + b

def parab(x, a, b, c):
    return a*x**2 + b*x + c

def cub(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d


def chi2(x, y, a, b, c, d):
    return sum((y[i] - (a*x[i]**3+b*x[i]**2+c*x[i]+d))**2 for i in range(len(x)))

def chi22(x, y, a, b, c):
    return sum((y[i] - (a*x[i]**2+b*x[i]+c))**2 for i in range(len(x)))


# ABERRATION ANALYSIS 
def spacingTrend(data):

    # create figure
    fig, ax1 = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(18.5, 12))
    fig.suptitle('Peak spacing over Position', fontsize = 22)

    par = []
    parr = []

    h = 0
    for i in range(3):
        for j in range(3):
            data1 = np.array(data[i+j+h])
            peakPositions = data1[:,0]
            peakSpacing = data1[:,1]
            DELTAX = peakPositions[-1] - peakPositions[0]
            XMIN = peakPositions[0] - DELTAX * 5/100
            XMAX = peakPositions[-1] + DELTAX * 5/100
            par1, _ = curve_fit(cub, peakPositions, peakSpacing)
            par2, _ = curve_fit(parab, peakPositions, peakSpacing)
            xs = np.linspace(XMIN, XMAX, 1000)
            # show plots
            ax1[i][j].plot(peakPositions, peakSpacing, marker = '.', markersize = 12, linewidth = 0, color = '#000000', label = 'data', zorder = 0)
            ax1[i][j].plot(xs, cub(xs, *par1), marker = '.', markersize = 0, linewidth = 2, linestyle = '-.', color = '#009AFF', label = 'cubic fit', zorder = 2)
            # ax2.plot(peakPositions, peakFWHM, marker = '.', markersize = 18, linewidth = 0, color = '#0451FF')
            ax1[i][j].plot(xs, parab(xs, *par2), marker = '.', markersize = 0, linewidth = 2, linestyle = 'dashed', color = '#FF4B00', label = 'quadratic fit', zorder = 1)
            # titles
            # ax1[i][j].set_title('Peak spacing over Position', fontsize = 22)
            # ax2.set_title('FWHM over Position', fontsize = 22)
            # labels
            # ax1[i][j].set_xlabel('Peak position [# pixel]', fontsize = 18)
            # ax2.set_xlabel('Peak position [# pixel]', fontsize = 18)
            # ax1[i][j].set_ylabel('Peak spacing [# pixel]', fontsize = 18)
            # ax2.set_ylabel('FWHM [# pixel]', fontsize = 18)
            ax1[i][j].tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5)
            # ax2.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 7)
            ax1[i][j].set_xlim(XMIN, XMAX)
            # ax2.set_xlim(XMIN, XMAX)
            ax1[i][j].legend(loc = 'upper right', prop = {'size': 12}, 
                ncol = 1, frameon = True, fancybox = False, framealpha = 1)
            nset = i+j+h+1
            # print('SET NUMBER ' + format(nset, '1.0f'))
            # print('Cubic fit:\t' + format(par1[0],'1.3e')+'*x^3 ' + format(par1[1],'1.3e')+'*x^2 ' + format(par1[2],'1.3e')+'*x')
            # print('Quadratic fit:\t' + format(par2[0],'1.3e')+'*x^2 ' + format(par2[1],'1.3e')+'*x\n')

            par.append(par1)
            parr.append(par2)
        h+=2

    fig.tight_layout()
    return fig, ax1, np.array(par), np.array(parr)



def main(argv):

    data = [readData(fname) for fname in filenames]
    fig, ax1, par, parr = spacingTrend(data)

    fig2, ax2 = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(18.5, 12))

    h = 0
    for i in range(3):
        for j in range(3):
            a = par[i+j+h][0]
            b = par[i+j+h][1]
            c = par[i+j+h][2]
            d = par[i+j+h][3]
            agrid = np.linspace(a-a/5, a+a/5, 500)
            bgrid = np.linspace(b-b/5, b+b/5, 500)
            cgrid = np.linspace(c-c/5, c+c/5, 500)
            dgrid = np.linspace(d-d/5, d+d/5, 500)

            aa, bb = np.meshgrid(agrid, bgrid)
            zz = chi2(np.array(data[i+j+h])[:,0], np.array(data[i+j+h])[:,1], aa, bb, c, d)

            ax2[i][j].contourf(agrid, bgrid, zz, levels = 100, cmap = 'rainbow')

            # bb, cc = np.meshgrid(bgrid, cgrid)
            # zz = chi2(np.array(data[i+j+h])[:,0], np.array(data[i+j+h])[:,1], a, bb, cc, d)

            # ax2[i][j].contourf(bgrid, cgrid, zz, levels = 100, cmap = 'rainbow')

        h+=2
    fig2.tight_layout()

    # fig3, ax3 = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(18.5, 12))

    # h = 0
    # for i in range(3):
    #     for j in range(3):
    #         a = parr[i+j+h][0]
    #         b = parr[i+j+h][1]
    #         c = parr[i+j+h][2]
    #         agrid = np.linspace(a-a/5, a+a/5, 500)
    #         bgrid = np.linspace(b-b/5, b+3*b, 500)
    #         cgrid = np.linspace(c-c/5, c+c/5, 500)

    #         aa, bb = np.meshgrid(agrid, bgrid)
    #         zz = chi22(np.array(data[i+j+h])[:,0], np.array(data[i+j+h])[:,1], aa, bb, c)

    #         ax3[i][j].contourf(agrid, bgrid, zz, levels = 100, cmap = 'rainbow')
    #     h+=2
    # fig3.tight_layout()


    fig.savefig(outputPath + 'grid' + '_trend.png', dpi = 500, facecolor = 'white')
    plt.show()

    return




if __name__ == "__main__":
    main(sys.argv[1:])