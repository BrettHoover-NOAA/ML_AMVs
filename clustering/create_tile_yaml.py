import numpy as np
import argparse
#
# define internal functions:
#
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
    parser.add_argument('nPreBins', metavar='PREBINS', type=int, help='number of pressure-bins')
    parser.add_argument('nTimBins', metavar='TIMBINS', type=int, help='number of time-bins')
    parser.add_argument('preMin', metavar='PREMIN', type=float, help='minimum pressure to tile (Pa)')
    parser.add_argument('preMax', metavar='PREMAX', type=float, help='maximum pressure to tile (Pa)')
    parser.add_argument('timMin', metavar='TIMMIN', type=float, help='minimum time to tile (frac. hrs)')
    parser.add_argument('timMax', metavar='TIMMAX', type=float, help='maximum time to tile (frac. hrs)')
    parser.add_argument('preHalo', metavar='PREHALO', type=float, help='extent of halo in pressure-dimension (Pa)')
    parser.add_argument('timHalo', metavar='TIMHALO', type=float, help='extent of halo in time-dimension (frac. hrs)')
    # parse arguments
    userInputs = parser.parse_args()
    # define some buffers around pressure and time to apply to the hard-limits prescribed by the user
    preBuffer = 2500.
    timBuffer = 0.
    # create pressure- and time-dimension bin edges based on user specifications, including buffers
    preBinEdges = np.linspace(start=userInputs.preMin - preBuffer, stop=userInputs.preMax + preBuffer, num=userInputs.nPreBins + 1)
    timBinEdges = np.linspace(start=userInputs.timMin - timBuffer, stop=userInputs.timMax + timBuffer, num=userInputs.nTimBins + 1)
    # run generate_tile_yaml() to report YAML as screen text
    generate_tile_yaml(preBinEdges, timBinEdges, userInputs.preHalo, userInputs.timHalo)
#
# end
#
