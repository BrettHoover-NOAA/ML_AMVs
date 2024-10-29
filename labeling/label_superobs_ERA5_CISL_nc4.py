# ML-labeling compliant
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import interpn
from os.path import isfile
import argparse
import datetime
#
# define internal functions
#
#
# INPUTS:
#   varName: name of variable to extract from ERA5 NC4 files (string)
#   xCoordName: name of coordinate-variable in x-direction in ERA5 NC4 files (string)
#   yCoordName: name of coordinate-variable in y-direction in ERA5 NC4 files (string)
#   zCoordName: name of coordinate-variable in z-direction in ERA5 NC4 files (string)
#   hrMinus3NC4File: full-path name of ERA5 NC4 file at -3 hours relative to analysis time (string)
#   hrMinus3NC4Hour: time-index for -3 hours in 24-hour hrMinus3NC4File (integer)
#   hrMinus2NC4File: full-path name of ERA5 NC4 file at -2 hours relative to analysis time (string)
#   hrMinus2NC4Hour: time-index for -2 hours in 24-hour hrMinus2NC4File (integer)
#   hrMinus1NC4File: full-path name of ERA5 NC4 file at -1 hours relative to analysis time (string)
#   hrMinus1NC4Hour: time-index for -1 hours in 24-hour hrMinus1NC4File (integer)
#   hrZeroNC4File: full-path name of ERA5 NC4 file at 0 hours relative to analysis time (string)
#   hrZeroNC4Hour: time-index for 0 hours in 24-hour hrZeroNC4File (integer)
#   hrPlus1NC4File: full-path name of ERA5 NC4 file at +1 hours relative to analysis time (string)
#   hrPlus1NC4Hour: time-index for +1 hours in 24-hour hrPlus1NC4File (integer)
#   hrPlus2NC4File: full-path name of ERA5 NC4 file at +2 hours relative to analysis time (string)
#   hrPlus23NC4Hour: time-index for +2 hours in 24-hour hrPlus2NC4File (integer)
#   hrPlus3NC4File: full-path name of ERA5 NC4 file at +3 hours relative to analysis time (string)
#   hrPlus3NC4Hour: time-index for +3 hours in 24-hour hrPlus3NC4File (integer)
#
# OUTPUTS:
#   zCoord: 1-D array in (nz,)-dimension containing values of coordinate in z-direction (numPy array)
#   yCoord: 1-D array in (ny,)-dimension containing values of coordinate in y-direction (numPy array)
#   xCoord: 1-D array in (nx,)-dimension containing values of coordinate in x-direction (numPy array)
#   tCoord: 1-D array in (nt,)-dimension containing values of coordinate in t-direction (numPy array)
#   cube4D: array in (nz,nxy,nx,nt)-dimension containing ERA5 data from varName variable (numPy array)
#
# DEPENDENCIES:
#
# numpy
# netCDF4.Dataset()
# os
def generate_ERA5_4dCube(varName, xCoordName, yCoordName, zCoordName, hrMinus3NC4File, hrMinus3NC4Hour,
                         hrMinus2NC4File,  hrMinus2NC4Hour, hrMinus1NC4File,  hrMinus1NC4Hour, hrZeroNC4File,
                         hrZeroNC4Hour, hrPlus1NC4File,  hrPlus1NC4Hour, hrPlus2NC4File,  hrPlus2NC4Hour,
                         hrPlus3NC4File, hrPlus3NC4Hour,):
    # extract coordinate values from hrZeroNC4File, or report error and return None values
    if isfile(hrZeroNC4File):
        nc4Hdl = Dataset(hrZeroNC4File)
        try:
            xCoord = np.asarray(nc4Hdl.variables[xCoordName]).squeeze()
            yCoord = np.asarray(nc4Hdl.variables[yCoordName]).squeeze()
            zCoord = np.asarray(nc4Hdl.variables[zCoordName]).squeeze()
            # tCoord is presumed to be np.arange(-3., 3.01, 1.), hourly files from -3 to +3 hours
            tCoord = np.arange(-3., 3.01, 1.)
        except:
            print('NC4 FILE: ' + hrZeroNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + hrZeroNC4File + ' does not exist')
        return None, None, None, None, None
    # define dimension-sizes
    ny = np.size(yCoord)
    nx = np.size(xCoord)
    nz = np.size(zCoord)
    nt = np.size(tCoord)
    # generate a NaN 4D-cube of dimension (nz,ny,nx,nt)
    cube4d = np.nan * np.ones((nz,ny,nx,nt))
    # loop through hours of observation-window (-3:+3), pick NC4 file
    for t in range(nt):
        # identify the appropriate ERA5 file, if it doesn't exist its time-level will remain NaN
        # if requesting an unknown time, raise error and loop (current time in loop will be NaN)
        if tCoord[t] == -3.:
            nc4File = hrMinus3NC4File
            nc4Hour = hrMinus3NC4Hour
        elif tCoord[t] == -2.:
            nc4File = hrMinus2NC4File
            nc4Hour = hrMinus2NC4Hour
        elif tCoord[t] == -1.:
            nc4File = hrMinus1NC4File
            nc4Hour = hrMinus1NC4Hour
        elif tCoord[t] == 0.:
            nc4File = hrZeroNC4File
            nc4Hour = hrZeroNC4Hour
        elif tCoord[t] == 1.:
            nc4File = hrPlus1NC4File
            nc4Hour = hrPlus1NC4Hour
        elif tCoord[t] == 2.:
            nc4File = hrPlus2NC4File
            nc4Hour = hrPlus2NC4Hour
        elif tCoord[t] == 3.:
            nc4File = hrPlus3NC4File
            nc4Hour = hrPlus3NC4Hour
        else:
            print('No file available for time T=', t)
        # fill cube4d at appropriate time-level, if nc4File exists and contains varName, otherwise
        # leave time-level NaN
        if isfile(nc4File):
            nc4Hdl = Dataset(nc4File)
            #try:
            a = np.asarray(nc4Hdl.variables[varName]).squeeze()
            cube4d[:,:,:,t] = a[nc4Hour,:,:,:].squeeze()
            #except:
            #    print('Failure to extract variable ' + varName + ' for time {:d}'.format(int(tCoord[t])))
    # Return coordinate vectors and 4D cube
    return zCoord, yCoord, xCoord, tCoord, cube4d


# duplicate_periodic_dimension_points: duplicate the data-columns along periodic dimensions to provide the periodic
# points on either side of the array along the periodic dimension. For example, for a 2D grid (lon,lat), the periodic
# dimension is the 0-dimension, and the values grid[0,:] will be duplicated at the end of the 0-dimension, and the values
# grid[-1,:] will be duplicated at the beginning of the 0-dimension. Returns the expanded array with duplicated points and
# an equivalent set of coordinate values with fictitious values for the duplicated book-ends. Presumes that the periodic
# dimension is equally-spaced.
#
# INPUTS:
#   inputArray: some input array that contains a periodic dimension (numPy array)
#   perDim: the dimension of inputArray that is periodic (integer)
#   perCoord: vector of coordinate-values along periodic dimension (numPy array)
#
# OUTPUTS:
#   outputArray: same as inputArray, but with duplicate points on either side of periodic dimension (numPy array)
#   perCoordNew: same as perCoord, but with fictitious coordinate-values for duplicate points on either side of dimension (numPy array)
#
# DEPENDENCIES:
#   numpy
def duplicate_periodic_dimension_points(inputArray, perDim, perCoord):
    # define dimension sizes, assert as array to make the values mutable
    dimSizes = list(np.shape(inputArray))
    # create new dimension sizes with dimSizes[perDim] inflated by 2
    dimSizesNew = dimSizes.copy()
    dimSizesNew[perDim] = dimSizesNew[perDim] + 2
    # create an output array with inflated dimension sizes, asserted as tuple
    outputArray = np.nan * np.ones(tuple(dimSizesNew))
    # create a copy of inputArray we can mutate
    inArray = np.copy(inputArray)
    # move perDim to axis=0
    inArray = np.moveaxis(inArray, perDim, 0)
    outputArray = np.moveaxis(outputArray, perDim, 0)
    # include inputArray as interior of outputArray
    outputArray[1:-1, ...] = inArray[...]
    # duplicate inArray[-1, ...] along outputArray[0, ...]
    outputArray[0, ...] = inArray[-1, ...]
    # duplicate inArray[0, ...] along outputArray[-1, ...]
    outputArray[-1, ...] = inArray[0, ...]
    # move axis=0 back to perDim
    outputArray = np.moveaxis(outputArray, 0, perDim)
    # redefine perCoord to book-end fictitious values of perCoord[0] - deltaCoord and perCoord[-1] + deltaCoord for periodic points
    deltaCoord = np.diff(perCoord)[0]  # all np.diff() values, the step between values of periodic coordinate, should be identical
    # create new perCoord with expanded dimension
    perCoordNew = np.nan * np.ones((np.size(perCoord)+2,))
    perCoordNew[1:-1] = perCoord
    perCoordNew[0] = perCoord[0] - deltaCoord
    perCoordNew[-1] = perCoord[-1] + deltaCoord
    # return outputArray and perCoordNew
    return outputArray, perCoordNew

# filter_obs_ERA5: Search through observational data and remove any observation that exists outside of the coordinate
# bounds of the ERA5 4d-cube. Observations past the edges of the periodic domain can be preserved if the 4d-cube is
# modified via duplicate_periodic_dimension_points() first.
#
# INPUTS:
#  xOb: vector of observation x-coordinate values (numpy array)
#  yOb: vector of observation y-coordinate values (numpy array)
#  zOb: vector of observation z-coordinate values (numpy array)
#  tOb: vector of observation t-coordinate values (numpy array)
#  xCoord: vector of ERA5 4d-cube x-coordinate values (numpy array)
#  yCoord: vector of ERA5 4d-cube y-coordinate values (numpy array)
#  zCoord: vector of ERA5 4d-cube z-coordinate values (numpy array)
#  tCoord: vector of ERA5 4d-cube t-coordinate values (numpy array)
#
# OUTPUT:
#   idxKeep: index of observations that exist within bounds of ERA5 4d-cube (numpy array)
#
# DEPENDENCIES:
#   numpy
def filter_obs_ERA5(xOb, yOb, zOb, tOb, xCoord, yCoord, zCoord, tCoord):
    x = np.where((tOb >= np.min(tCoord)) &
                 (tOb <= np.max(tCoord)) &
                 (zOb >= np.min(zCoord)) &
                 (zOb <= np.max(zCoord)) &
                 (xOb >= np.min(xCoord)) &
                 (xOb <= np.max(xCoord)) &
                 (yOb >= np.min(yCoord)) &
                 (yOb <= np.max(yCoord)) 
                )[0]
    return x
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directories and names of ' +
                                                 'netCDF super-ob and ERA5 NC4 files')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    parser.add_argument('obDataDir', metavar='OBDATADIR', type=str, help='full path to data directory for super-observations')
    parser.add_argument('inputNetcdfFile', metavar='INFILE', type=str, help='input netCDF file (unlabeled super-obs)')
    parser.add_argument('era5DataDir', metavar='ERA5DATADIR', type=str, help='full path to data directory for ERA5')
    parser.add_argument('outputNetcdfFile', metavar='OUTFILE', type=str, help='output netCDF file (superobs with metadata)')
    # parse arguments
    userInputs = parser.parse_args()
    # quality-control inputs:
    # if userInputs.obDataDir does not end in '/', append it
    obDataDir = userInputs.obDataDir + '/' if userInputs.obDataDir[-1] != '/' else userInputs.obDataDir
    # if userInputs.era5DataDir does not end in '/', append it
    era5DataDir = userInputs.era5DataDir + '/' if userInputs.era5DataDir[-1] != '/' else userInputs.era5DataDir
    # define analysis datetime
    anaDateTime = datetime.datetime.strptime(userInputs.anaDateTime, '%Y%m%d%H')
    # define ERA5 daily NC4 files for each hour of observation-window, and define the time-index of the
    # hour in the daily NC4 file
    # minus 3 hours
    era5DateTime = anaDateTime - datetime.timedelta(hours=3)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrM3FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM3FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM3Hour = int(HH)
    # minus 2 hours
    era5DateTime = anaDateTime - datetime.timedelta(hours=2)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrM2FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM2FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM2Hour = int(HH)
    # minus 1 hours
    era5DateTime = anaDateTime - datetime.timedelta(hours=1)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrM1FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM1FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrM1Hour = int(HH)
    # zero hours (analysis time)
    era5DateTime = anaDateTime
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hr00FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hr00FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hr00Hour = int(HH)
    # plus 1 hours
    era5DateTime = anaDateTime + datetime.timedelta(hours=1)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrP1FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP1FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP1Hour = int(HH)
    # plus 2 hours
    era5DateTime = anaDateTime + datetime.timedelta(hours=2)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrP2FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP2FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP2Hour = int(HH)
    # plus 3 hours
    era5DateTime = anaDateTime + datetime.timedelta(hours=3)
    YYYY = era5DateTime.strftime('%Y')
    MM = era5DateTime.strftime('%m')
    DD = era5DateTime.strftime('%d')
    HH = era5DateTime.strftime('%H')
    hrP3FileU = era5DataDir + 'e5.oper.an.pl.128_131_u.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP3FileV = era5DataDir + 'e5.oper.an.pl.128_132_v.ll025uv.' + YYYY + MM + DD + '00_' + YYYY + MM + DD + '23.nc'
    hrP3Hour = int(HH)
    # generate ERA5 4d-cube for u- and v-component of wind
    # coordinate values are the same for both variables, so we only bother to define them for um
    zCoord, yCoord, xCoord, tCoord, um = generate_ERA5_4dCube('U', 'longitude', 'latitude', 'level', hrM3FileU, hrM3Hour,
                                                              hrM2FileU, hrM2Hour, hrM1FileU, hrM1Hour, hr00FileU, hr00Hour, hrP1FileU,
                                                              hrP1Hour, hrP2FileU, hrP2Hour, hrP3FileU, hrP3Hour)
    vm = generate_ERA5_4dCube('V', 'longitude', 'latitude', 'level', hrM3FileV, hrM3Hour, hrM2FileV, hrM2Hour, hrM1FileV,
                              hrM1Hour, hr00FileV, hr00Hour, hrP1FileV, hrP1Hour, hrP2FileV, hrP2Hour, hrP3FileV, hrP3Hour)[4]
    # add duplicated points on periodic (longitude) dimension of 4d-cubes
    # new coordinate values are the same for both variables, so we only bother to define them for um
    um2, xCoordNew = duplicate_periodic_dimension_points(um, 2, xCoord)
    vm2 = duplicate_periodic_dimension_points(vm, 2, xCoord)[0]
    # extract superOb data from SOFile
    obHdl = Dataset(obDataDir + userInputs.inputNetcdfFile)
    SOlat = np.asarray(obHdl.variables['lat']).squeeze()
    SOlon = np.asarray(obHdl.variables['lon']).squeeze()  # -180 to 180 degree format, needs to be transformed to 0 to 360 degree format
    SOpre = np.asarray(obHdl.variables['pre']).squeeze()  # Pa, needs to be transformed to hPa
    SOtim = np.asarray(obHdl.variables['tim']).squeeze()
    SOuwd = np.asarray(obHdl.variables['uwd']).squeeze()
    SOvwd = np.asarray(obHdl.variables['vwd']).squeeze()
    # filter to include only those obs inside ERA5 4d-cube
    x = np.copy(SOlon)
    fix = np.where(x<0.)[0]
    x[fix] = x[fix] + 360.
    SOkp = filter_obs_ERA5(x, SOlat, 0.01*SOpre, SOtim, xCoordNew, yCoord, zCoord, tCoord)
    # define 4d-cube dimensions as (log-10(z), y, x, t) and perform multi-dimensional linear interpolation
    # to observation (log-10(z), y, x, t) values to perform log-interpolation of pressure in vertical and
    # linear interpolation along all other dimensions)
    dims4dCube = (np.log10(zCoord), yCoord, xCoordNew, tCoord)  # set of 4d-cube dimensions for linear interpolation
    # copy equivalent coordinates from observations to new arrays - this is done because some of the coordinate values
    # need to be transformed into the same units as the 4d-cube, but we don't want to mutate the obs data directly.
    obPre = 0.01 * SOpre
    obLat = np.copy(SOlat)
    obLon = np.copy(SOlon)
    fix = np.where(obLon < 0.)
    obLon[fix] = obLon[fix] + 360.
    obTim = SOtim
    # interpolate the um and vm 4d-cube to the observation coordinates
    ERA5uwd = interpn(dims4dCube, um2, np.asarray([np.log10(obPre[SOkp]), obLat[SOkp], obLon[SOkp], obTim[SOkp]]).T)
    ERA5vwd = interpn(dims4dCube, vm2, np.asarray([np.log10(obPre[SOkp]), obLat[SOkp], obLon[SOkp], obTim[SOkp]]).T)
    # generate output arrays of the same size as observation vectors, with NaN values outside of SOkp and interpolated
    # ERA5 wind values at SOkp points, to store labels
    SOlabel_u = np.nan * np.ones(np.shape(SOlat))
    SOlabel_v = np.nan * np.ones(np.shape(SOlat))
    SOlabel_u[SOkp] = ERA5uwd
    SOlabel_v[SOkp] = ERA5vwd
    # write labels to output file
    ncOutFileName = obDataDir + userInputs.outputNetcdfFile
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
    # add variables
    uwd = ncOut.createVariable(
                                  'uwd'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    vwd = ncOut.createVariable(
                                  'vwd'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    # assign netCDF file attributes
    ncOut.labelSource = ("ERA5 hourly data on pressure levels 1940 to present, reanalysis zonal and meridional wind on 0.25 " + 
                         "degree global grids on 37 pressure levels, via Copernicus Licence " + 
                         "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview " +
                         "and interpolated linearly in latitude, longitude, and time, and interpolated log10-linearly in pressure, " +
                         "to observation's location in space and time")
    ncOut.labelCitation = ("Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., " +
                           "Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. " +
                           "(2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service " +
                           "(C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47")
    ncOut.analysisDateTime = userInputs.anaDateTime
    ncOut.sourceFile = userInputs.inputNetcdfFile
    # fill netCDF file variables with labels
    uwd[:]      = SOlabel_u
    vwd[:]      = SOlabel_v
    # close ncOut
    ncOut.close()
#
# end
#
