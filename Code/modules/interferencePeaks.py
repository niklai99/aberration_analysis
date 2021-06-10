import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# constants
LAMBDA = 585.3 # nanometers
d = 3.47 * 1e6 #nanometers (3.47 millimiters)
dataPath = '../Data/'

start = 0
end = 7926


binFrac=3 # nBins_new = nBins_old / binFrac




# read data from txt file
def readData(fname):
    # read raw data
    data = pd.read_csv(dataPath + fname, sep = '\t', header = None, names = ['X', 'Y'])

    # change bins and get new y
    newY, edge= np.histogram(data.X, weights=data.Y, bins=int(len(data.X)/binFrac))

    # get new x
    newX = []
    for i in range(len(edge)-1):
        newX.append((edge[i]+edge[i+1])/2)

    # save new data in a Pandassssss dataframe
    data = pd.DataFrame(list(zip(newX,newY)), columns=['X','Y'])

    return data





def plotRawPeaks(newData):
    fig, ax = plt.subplots(figsize=(14,8))

    # plot data
    ax.hist(newData['X'], bins = int(len(newData['X'])), weights = newData['Y'], histtype = 'step', color = '#0451FF', linewidth = 1.5)
    
    ax.set_xlim(start, end)
    ax.set_ylim(0, np.amax(newData['Y']) * ( 1 + 5/100 ))
    
    ax.set_title('Interference Peaks', fontsize = 24)
    ax.set_xlabel('# pixel', fontsize = 20)
    ax.set_ylabel('ADC counts', fontsize = 20, loc = 'top')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16, direction = 'out', length = 10)
    ax.set_ylim(top = 1600)
    
    return fig, ax




# fitting function
def Gauss(x, N, x0, sigma):
    return N/(2*np.pi)**0.5 / sigma * np.exp(-(x - x0)**2 / (2 * sigma**2))




def findPeaks(newData):

    # use scipy signal to identify peaks
    peaks, p = find_peaks(newData['Y'], width=5, prominence=100, rel_height=0.5)
    print("Found", len(peaks), "peaks")

    fig, ax = plt.subplots(figsize=(14,8))

    # plot data
    ax.hist(newData['X'], bins = int(len(newData['X'])), weights = newData['Y'], histtype = 'step', color = '#0451FF', linewidth = 1.5)


    xHalfLeft = []
    xHalfRight = []
    xPlotLeft = []
    xPlotRight = []
    FWHM = []
    Peaks = []
    parFit=[]

    
    # loop over peaks
    for i in range(len(peaks)):
        # get fit data
        tr=int((p['right_ips'][i] - p['left_ips'][i])/5)
        xfit=newData['X'].iloc[round(p['left_ips'][i]-tr):round(p['right_ips'][i])+tr]
        yfit=newData['Y'].iloc[round(p['left_ips'][i]-tr):round(p['right_ips'][i])+tr]

        # initial parameters
        mean0=np.average(xfit)
        std0=np.std(xfit)

        # normalization parameter
        ngau=np.sum(yfit*binFrac)

        # fit current peak
        par, cov = curve_fit(lambda x,mean,stddev: Gauss(x,ngau,mean,stddev),
                             xfit, yfit,
                             p0=[mean0, std0])


        halfMax = Gauss(par[0],ngau,*par) / 2

        # compute halfMax pixels
        spline = UnivariateSpline(xfit, Gauss(xfit,ngau,*par) - halfMax , s = 0)
        r1, r2 = spline.roots() # find the roots
        

        # ax.vlines(x=r1, ymin=0, ymax = np.amax(newData['Y']) * ( 1 + 5/100 ), color = "blue", alpha = 0.3)
        # ax.vlines(x=r2, ymin=0, ymax = np.amax(newData['Y']) * ( 1 + 5/100 ), color = "blue", alpha = 0.3)

        FWHM.append(r2-r1)
        Peaks.append(par[0])
        xHalfLeft.append(r1)
        xHalfRight.append(r2)
        xPlotLeft.append(par[0] - 2*(r2-r1))
        xPlotRight.append(par[0] + 2*(r2-r1))

        # chi2
        diff = yfit-Gauss(xfit,ngau,*par)
        chisq = np.sum(diff**2)/(len(xfit)-2)
        #print("chisq",i, round(abs(chisq-(len(xfit-2)))/ (len(xfit)-2)**0.5 / 2**0.5,2)) # TODO: ricontrollami
        
        # plotting data
        xth = np.linspace(xPlotLeft, xPlotRight, 100)
        yth = Gauss(xth,ngau,*par)

        # plot gaussian fit
        ax.plot(xth, yth, color='#FF4B00', alpha = 0.8, linestyle = 'dashed')

        # save parameters
        parFit.append([ngau, par[0], par[1]])


    return Peaks, FWHM, xHalfLeft, xHalfRight, xPlotLeft, xPlotRight, parFit, ax, fig