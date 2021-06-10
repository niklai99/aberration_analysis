import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

dataPath = '../Data/'
outputPath = '../Plots/interference/'


binFrac=3 # nBins_new = nBins_old / binFrac

beg = 0
stp = 7926


# read data from txt file
def readData(fname):
    # read raw data
    data = pd.read_csv(dataPath + fname + '_proj.txt', sep = '\t', header = None, names = ['X', 'Y'])

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
    
    ax.set_xlim(np.amin(newData['X']), np.amax(newData['X']))
    ax.set_ylim(0, np.amax(newData['Y']) * ( 1 + 5/100 ))
    
    ax.set_title('Interference Peaks', fontsize = 22)
    ax.set_xlabel('# pixel', fontsize = 18)
    ax.set_ylabel('ADC counts', fontsize = 18, loc = 'top')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 7)
 
    return fig, ax




# fitting function
def Gauss(x, N, x0, sigma):
    return N/(2*np.pi)**0.5 / sigma * np.exp(-(x - x0)**2 / (2 * sigma**2))


def findPeaks(newData):

    # use scipy signal to identify peaks
    peaks, p = find_peaks(newData['Y'], width=5, prominence=100, rel_height=0.7)
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
        
        # ax.vlines(x=par[0], ymin=0, ymax = np.amax(newData['Y']) * ( 1 + 5/100 ), color = "red", alpha = 0.3)
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

    ax.set_xlim(np.amin(newData['X']), np.amax(newData['X']))
    ax.set_ylim(0, np.amax(newData['Y']) * ( 1 + 5/100 ))
    
    ax.set_title('Interference Peaks', fontsize = 22)
    ax.set_xlabel('# pixel', fontsize = 18)
    ax.set_ylabel('ADC counts', fontsize = 18, loc = 'top')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'out', length = 7)

    return Peaks, FWHM, xHalfLeft, xHalfRight, xPlotLeft, xPlotRight, parFit, ax, fig




# function to select n elements and skip m elements
def select_skip(iterable, select, skip):
    return [x for i, x in enumerate(iterable) if i % (select+skip) < select]


def computeSpacing(Peaks):

    C = []
    Spacing = []

    for i in range(1, len(Peaks)):
        C.append(Peaks[i] - Peaks[i-1])

    for i in range(1, len(C)):
        Spacing.append((C[i] + C[i-1]) / 2)

    return Spacing



def main(argv):

    fname = argv[0]
    # read data from file
    data = readData(fname)

    if len(argv)>=2 and argv[1] == 'r':
        fig, ax = plotRawPeaks(data)
    
    else:
        if len(argv) >= 3:
            start = int(argv[1])
            end = int(argv[2])
        else:
            start = beg
            end = stp


        # select data based on requested X range
        newData = data[(data['X']>start) & (data['X']<end)]
        newData.reset_index(inplace = True, drop = True)



        # find peaks
        Peaks, FWHM, xHalfLeft, xHalfRight, xPlotLeft, xPlotRight, parFit, ax, fig = findPeaks(newData)

        # compute spacing between peaks
        Spacing = computeSpacing(Peaks)


        if len(argv) == 4:
            oname = argv[3]
            # save data for trends
            np.savetxt(dataPath + oname + '.txt', np.c_[Peaks[1:-1], Spacing, FWHM[1:-1]], delimiter=';')


    fig.tight_layout()
    fig.savefig(outputPath + fname + '_interference.png', dpi = 500, facecolor = 'white')
    plt.show()

    return




if __name__ == "__main__":
    main(sys.argv[1:])
