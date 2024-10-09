# collects CNN 2D subgrids distributed on pressure-time tiles and creates a single file with CNN grids in the same
# order as observations along the ob-dimension
# import modules
import numpy as np
from netCDF4 import Dataset
from glob import glob
import argparse
#
# begin
#
if __name__ == "__main__":
    # generate an argument parser to accept command-line inputs:
    parser = argparse.ArgumentParser(description='Define search-string for tiles, and output file-name')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full-path to data directory (ends in /)')
    parser.add_argument('searchString', metavar='SRCHSTR', type=str, help='search string up to but not including wildcard for locating tiles')
    parser.add_argument('outputNetcdfFileName', metavar='OUTFILE', type=str, help='name of output netCDF file')
    parser.add_argument('numTiles', metavar='NUMTILES', type=str, help='number of tiles that should be processed')
    userInputs = parser.parse_args()
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    # assert numTiles as integer
    numTiles = int(userInputs.numTiles)
    # find list of tile files to process
    tileList = glob(dataDir + userInputs.searchString + '*' + '.nc')
    tileList.sort()
    # check if tileList is the correct length, if tiles are missing report and exit
    if len(tileList) != numTiles:
        print('ERROR: NUMBER OF TILE FILES {:d} DOES NOT MATCH EXPECTED NUMBER OF TILES {:d}, ABORTING'.format(len(tileList), numTiles))
        raise SystemExit('num-tile mismatch error')
    print('processing {:d} tiles'.format(len(tileList)))
    # scan through all tiles to determine final array-size of output
    # all files should contain identical (x,y) dimension-sizes, so we will just read the first file for these
    hdl = Dataset(tileList[0])
    nx = hdl.dimensions['x'].size
    ny = hdl.dimensions['y'].size
    # accumulate ob-dimension across all tiles
    nOb = 0
    for tile in tileList:
        hdl = Dataset(tile)
        nOb = nOb + hdl.dimensions['ob'].size
    print('constructing ({:d},{:d},{:d}) sized CNN output'.format(nOb, ny, nx))
    CNN_data = np.zeros((nOb,ny,nx))
    lat_data = np.zeros((nOb,ny))
    lon_data = np.zeros((nOb,nx))
    # scan through all tiles to populate CNN_data with subgrid data
    for tile in tileList:
        print('processing ' + tile)
        hdl = Dataset(tile)
        # obtain index values
        idx = np.asarray(hdl.variables['idx']).squeeze()
        # obtain CNN subgrid values (2D temperature)
        fld = np.asarray(hdl.variables['T']).squeeze()
        # uncomment following lines to carry through lat/lon data (if it is being collected in Tiles!)
        #la = np.asarray(hdl.variables['lat']).squeeze()
        #lo = np.asarray(hdl.variables['lon']).squeeze()
        # /uncomment
        # populate appropriate portions of CNN_data
        CNN_data[idx,:,:] = fld
        # uncomment following lines to carry through lat/lon data (if it is being collected in Tiles!)
        #lat_data[idx,:] = la
        #lon_data[idx,:] = lo
        # /uncomment
    # write to output file
    # output to netCDF
    ncOutFileName = dataDir + userInputs.outputNetcdfFileName
    ncOut = Dataset(
                      ncOutFileName  , # Dataset input: Output file name
                      'w'              , # Dataset input: Make file write-able
                      format='NETCDF4' , # Dataset input: Set output format to netCDF4
                    )
    # add dimensions
    ob  = ncOut.createDimension(
                                 'ob' , # nc_out.createDimension input: Dimension name
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    x  = ncOut.createDimension(
                                 'x' , # nc_out.createDimension input: Dimension name
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    y  = ncOut.createDimension(
                                 'y' , # nc_out.createDimension input: Dimension name
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    # add variables
    # uncomment the lines below to track lat/lon on the edge of each subgrid as well (turned off to save on storage space
    #lat = ncOut.createVariable(
    #                              'lat'       ,
    #                              'f8'        ,
    #                              ('ob','y')
    #                            )
    #lon = ncOut.createVariable(
    #                              'lon'       ,
    #                              'f8'        ,
    #                              ('ob','x')
    #                            )
    # /uncomment
    CNN = ncOut.createVariable(
                                  'CNN'       ,
                                  'f8'        ,
                                  ('ob', 'y', 'x')
                                )
    # write to file
    CNN[:,:,:] = CNN_data
    # uncomment the lines below to track lat/lon on the edge of each subgrid as well (turned off to save on storage space
    #lat[:,:] = lat_data
    #lon[:,:] = lon_data
    # /uncomment
    # close netCDF file
    ncOut.close()
#
# end
#
