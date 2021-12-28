import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# constants
dataPath = '../Data/'
outputPath = '../Plots/trends/'


# read data from txt file
def readData(fname):

    return np.loadtxt(dataPath + fname + '_data.txt', delimiter=';')


def lin(x, a, b):
    return a*x + b

def parab(x, a, b, c):
    return a*x**2 + b*x + c

def cub(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d


# ABERRATION ANALYSIS 
def spacingTrend(peakPositions, peakSpacing, peakFWHM):

    DELTAX = peakPositions[-1] - peakPositions[0]
    XMIN = peakPositions[0] - DELTAX * 5/100
    XMAX = peakPositions[-1] + DELTAX * 5/100

    par1, _ = curve_fit(cub, peakPositions, peakSpacing)
    par2, _ = curve_fit(lin, peakPositions, peakFWHM)

    # create figure
    fig = plt.figure(figsize=(16,8))

    # create axes
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    xs = np.linspace(XMIN, XMAX, 1000)

    # show plots
    ax1.plot(peakPositions, peakSpacing, marker = '.', markersize = 18, linewidth = 0, color = '#0451FF')
    ax1.plot(xs, cub(xs, *par1), marker = '.', markersize = 0, linewidth = 2, linestyle = 'dashed', color = '#FF4B00')
    ax2.plot(peakPositions, peakFWHM, marker = '.', markersize = 18, linewidth = 0, color = '#0451FF')
    ax2.plot(xs, lin(xs, *par2), marker = '.', markersize = 0, linewidth = 2, linestyle = 'dashed', color = '#FF4B00')

    # titles
    ax1.set_title('Peak spacing over Position', fontsize = 22)
    ax2.set_title('FWHM over Position', fontsize = 22)

    # labels
    ax1.set_xlabel('Peak position [# pixel]', fontsize = 18)
    ax2.set_xlabel('Peak position [# pixel]', fontsize = 18)
    ax1.set_ylabel('Peak spacing [# pixel]', fontsize = 18)
    ax2.set_ylabel('FWHM [# pixel]', fontsize = 18)

    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 7)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 7)

    ax1.set_xlim(XMIN, XMAX)
    ax2.set_xlim(XMIN, XMAX)

    fig.tight_layout()
    return fig, ax1, ax2



def main(argv):

    if len(argv) != 0:
        fname = argv[0]
        # read data from file
        data = readData(fname)
        fig, ax1, ax2 = spacingTrend(data[:,0], data[:,1], data[:,2])
        fig.savefig(outputPath + fname + '_trend.png', dpi = 500, facecolor = 'white')

    else:
        for i in range(4, 10):
            fname = 'scan_585_z1_0%s' % (i)
            data = readData(fname)
            fig, ax1, ax2 = spacingTrend(data[:,0], data[:,1], data[:,2])
            fig.savefig(outputPath + fname + '_trend.png', dpi = 500, facecolor = 'white')


    # plt.show()

    return




if __name__ == "__main__":
    main(sys.argv[1:])
