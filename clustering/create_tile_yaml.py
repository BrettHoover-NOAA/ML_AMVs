import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import argparse
#
# define internal functions:
#
# compute_equal_ob_density_bin_edges: Given the number of bins, a minimum and maximum value to tile across,
#                                     and the (very small) step-size used for searching, compute bin-edges
#                                     that produce the requested number of bins between the min/max values
#                                     that contain roughly an equal number of observations per-bin.
#
# INPUTS:
#    obVals: the observed values in the dimension to be tiled across
#    nBins: the number of bins to produce
#    minVal: the minimum value of the variable to tile across
#    maxVal: the maximum value of the variable to tile across
#    stepSize: the (very small) step-size used to search for optimal bin edges
#
# OUTPUTS:
#    equBinEdges: bin-edges spaced to optimize equal ob-density between bins
#
# DEPENDENCIES
#    numpy
#    scipy.interpolate.interp1d
def compute_equal_ob_density_bin_edges(obVals, nBins, minVal, maxVal, stepSize):
    # define bin-edges for incredibly small bins of size stepSize across total tiled space
    delVal = maxVal - minVal
    miniBinEdges = np.linspace(minVal, maxVal, int(np.round(delVal/stepSize)))
    # compute number of obs per mini-bin
    numObsPerMiniBin = np.nan * np.ones((np.size(miniBinEdges)-1,))
    for i in range(np.size(numObsPerMiniBin)):
        vMin = miniBinEdges[i]
        vMax = miniBinEdges[i+1]
        j = np.where((obVals >= vMin) & (obVals < vMax))
        numObsPerMiniBin[i] = np.size(j)
    # define cumulative percentage of total ob-count across mini-bins
    cumuP = np.cumsum(numObsPerMiniBin) / np.sum(numObsPerMiniBin)
    # define a target percentage of observations to capture in each bin
    pTarget = float(nBins)**-1
    # create an interpolator that computes the cumulative percentage of obs as a
    # function of the index of the mini-bin
    f = interp1d(cumuP, np.arange(np.size(cumuP)))
    # interpolate to regular intervals of pTarget to find equal-ob-density accumulation
    # of obs within accumulated mini-bins
    equMiniBinIndices=f(pTarget * np.arange(1, nBins+0.01))
    # find miniBinEdges values of these accumulation-points to define equal-ob-density bin edges
    equBinEdges=[minVal]
    for i in range(nBins):
        equBinEdges.append(miniBinEdges[int(np.round(equMiniBinIndices[i]))])
    # redefine the last bin-edge by the specified max value (it may have drifted slightly)
    equBinEdges[-1] = maxVal
    # return numpy vector of equBinEdges
    return np.asarray(equBinEdges).squeeze()


# generate_tile_yaml: given information on pressure- and time-dimension bin edges for tiles
#                     and halo size, generate a YAML specifying each tile for assign_AMV_clusters.py
# INPUTS:
#    preBinEdges: numpy vector of bin-edges for pressure-dimension of tiles (float, Pa)
#    timBinEdges: numpy vector of bin-edges for tim-dimension of tiles (float, fractional hours)
#    preHalo: size of halo extending from tile in pressure-dimension (Pa)
#    timHalo: size of halo extending from tile in time-dimension (Pa)
#
# OUTPUTS:
#    prints properly formatted YAML to screen
#
# DEPENDENCIES:
#    numpy
def generate_tile_yaml(preBinEdges, timBinEdges, preHalo, timHalo):
    nPreBins = np.size(preBinEdges) - 1
    nTimBins = np.size(timBinEdges) - 1
    nTiles = nPreBins * nTimBins
    n = 0
    for p in range(nPreBins):
        for t in range(nTimBins):
            n += 1
            print('Tile_{:03d}:\n'.format(n) +
                  '   tileValue: {:d}\n'.format(n) +
                  '   tilePressureMin: {:.7f}\n'.format(preBinEdges[p]) +
                  '   tilePressureMax: {:.7f}\n'.format(preBinEdges[p+1]) +
                  '   tileTimeMin: {:.7f}\n'.format(timBinEdges[t]) +
                  '   tileTimeMax: {:.7f}\n'.format(timBinEdges[t+1]) +
                  '   haloPressureMin: {:.7f}\n'.format(preBinEdges[p] - preHalo) +
                  '   haloPressureMax: {:.7f}\n'.format(preBinEdges[p+1] + preHalo) +
                  '   haloTimeMin: {:.7f}\n'.format(timBinEdges[t] - timHalo) +
                  '   haloTimeMax: {:.7f}'.format(timBinEdges[t+1] + timHalo)
                 )
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define tile and halo specifications')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full path to data directory')
    parser.add_argument('netcdfFileName', metavar='INFILE', type=str, help='netCDF AMV file')
    parser.add_argument('nPreBins', metavar='PREBINS', type=int, help='number of pressure-bins')
    parser.add_argument('preMin', metavar='PREMIN', type=float, help='minimum pressure to tile (Pa)')
    parser.add_argument('preMax', metavar='PREMAX', type=float, help='maximum pressure to tile (Pa)')
    parser.add_argument('preHalo', metavar='PREHALO', type=float, help='extent of halo in pressure-dimension (Pa)')
    parser.add_argument('nTimBins', metavar='TIMBINS', type=int, help='number of time-bins')
    parser.add_argument('timMin', metavar='TIMMIN', type=float, help='minimum time to tile (frac. hrs)')
    parser.add_argument('timMax', metavar='TIMMAX', type=float, help='maximum time to tile (frac. hrs)')
    parser.add_argument('timHalo', metavar='TIMHALO', type=float, help='extent of halo in time-dimension (frac. hrs)')
    parser.add_argument('optBins', metavar='OPTIMIZE', type=str, help='True==optimize tiles for even ob-density, False==regularized tiles')
    # parse arguments
    userInputs = parser.parse_args()
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    # specify netCDF file-handle and extract pressure and time vectors
    netcdfFileName = userInputs.netcdfFileName
    hdl=Dataset(dataDir + netcdfFileName)
    obPre=np.asarray(hdl.variables['pre']).squeeze()
    obTim=np.asarray(hdl.variables['tim']).squeeze()
    # create pressure- and time-dimension bin edges based on user specifications
    # userInputs.optBins==True will use optimization to generate tiles with roughly equal ob-density
    if (userInputs.optBins == 'True') | (userInputs.optBins == 'true') | (userInputs.optBins == 'TRUE') |\
       (userInputs.optBins == '.True.') | (userInputs.optBins == '.true.') | (userInputs.optBins == '.TRUE.'):
        # search for optimal bin-edges in pressure-dimension by 100 Pa (1 hPa) steps and in time-dimension
        # by 0.0166667 fractional hourly (~ 1 minute) steps
        preBinEdges = compute_equal_ob_density_bin_edges(obPre, userInputs.nPreBins, userInputs.preMin, userInputs.preMax, 100.)
        timBinEdges = compute_equal_ob_density_bin_edges(obTim, userInputs.nTimBins, userInputs.timMin, userInputs.timMax, 0.0166667)
    else:
        preBinEdges = np.linspace(start=userInputs.preMin, stop=userInputs.preMax, num=userInputs.nPreBins + 1)
        timBinEdges = np.linspace(start=userInputs.timMin, stop=userInputs.timMax, num=userInputs.nTimBins + 1)
    # run generate_tile_yaml() to report YAML as screen text
    generate_tile_yaml(preBinEdges, timBinEdges, userInputs.preHalo, userInputs.timHalo)
#
# end
#
