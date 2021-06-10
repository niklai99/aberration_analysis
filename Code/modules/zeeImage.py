import struct
import numpy as np

# constants
npixels = 7926

# read data from file and compute X-projection
def getData(fname, dataPath):

    # read binary file
    pf = open(dataPath + fname, "rb")

    # read data by chunks and get ncolumns
    ncolumns = 0
    byte = pf.read(npixels*2)
    while byte:
        ncolumns+=1
        byte = pf.read(npixels*2)
    pf.close()

    print("number of columns", ncolumns)

    # read file by chunks (again) to fill histogram
    pf = open(dataPath + fname, "rb")

    zhist = np.empty([npixels, ncolumns]) # array to store bin height for 2D hist
    projx = np.empty([ncolumns]) # array to store projection on the X axis

    byte = pf.read(npixels*2) # list to store npixels*2 bits of raw data
    icol=0
    while byte:
        sumpx=0
        intList=[] # list to store npixels short integers (2 bits)
        # loop over bits and ~unpack~ them into short integers
        for i in range(0, len(byte), 2):
            intList.append(struct.unpack('<h', byte[i:i+2])[0])
        # store computed integers in 2D matrix
        for i in range(npixels):
            zhist[i, icol] = intList[i]
            # get X projection
            sumpx += intList[i]

        # read next line and update counters
        projx[icol] = sumpx
        icol += 1
        byte = pf.read(npixels*2)

    pf.close()

    return zhist, ncolumns, projx
    

# compute and subtract background
def doBackgroundOp(bkgfrom, bkgto, hist):

    print("subtracting background from ", bkgfrom , "to", bkgto)
    if(bkgto >= len(hist[0])-1): bkgto=len(hist[0])-2

    ncolumns=np.shape(hist)[1]
    for i in range(npixels):

        bkg=0
        # compute mean background for row i
        for j in range(bkgfrom-1, bkgto):
            bkg += hist[i][j+1]
        bkg /= (bkgto-bkgfrom+1)
        #print(bkg)

        # subtract background from row i
        for j in range(ncolumns):
                hist[i][j] -= bkg

    # update X-projection
    projx = np.empty([ncolumns]) # array to store projection on the X axis
    for i in range(ncolumns):
        sumpx = 0
        for j in range(npixels):
            sumpx += hist[j][i]
        projx[i] = sumpx

    return hist, projx


# compute Y-projection
def projectToY(zhist, projYfrom, projYto, ncolumns):

    projy = np.empty(npixels)
    print("projecting from", projYfrom, projYto)

    for i in range(npixels):
        sumy = 0
        for j in range(projYfrom, projYto):
            sumy+=zhist[i][j]
        projy[i]=sumy

    return projy