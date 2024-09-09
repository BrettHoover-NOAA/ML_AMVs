# for documentation on EMCWF model-levels and sigma vs pressure values:
#
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height
#
# full model-levels are defined at the midpoint between the half-level surfaces, which may explain why a
# full model-level is nearly but not quite entirely a constant sigma-surface
# import modules
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import interpn
import datetime
from os.path import isfile
from time import time
import argparse
#
# define internal functions
#
# generate_ERA5_3dCube: append netCDF 2D fields across different times into a single 3D cube
# INPUTS:
#   varName: name of variable to extract from ERA5 GRIB files (string)
#   xCoordName: name of coordinate-variable in x-direction in ERA5 NC4 files (string)
#   yCoordName: name of coordinate-variable in y-direction in ERA5 NC4 files (string)
#   tCoordName: name of coordinate-variable in t-direction in ERA5 NC4 files (string)
#   beforeNC4File: full-path name of ERA5 NC4 file at "before" period relative to analysis time-window (string)
#   duringNC4File: full-path name of ERA5 NC4 file at "during" period relative to analysis time-window (string)
#   afterNC4File: full-path name of ERA5 NC4 file at "after" period relative to analysis time-window (string)
#   anaDateTime: date-time of analysis time (datetime.datetime() object)
#   tVals: vector of values along tCoord to extract into 3dcube (float vector)
#
# OUTPUTS:
#   yCoord: 1-D array in (ny,)-dimension containing values of coordinate in y-direction (numPy array)
#   xCoord: 1-D array in (nx,)-dimension containing values of coordinate in x-direction (numPy array)
#   tCoord: 1-D array in (nt,)-dimension containing values of coordinate in t-direction (numPy array)
#   cube3D: array in (ny,nx,nt)-dimension containing ERA5 data from varName variable (numPy array)
#
# DEPENDENCIES:
#
# numpy
# netCDF4.Dataset()
# datetime
# os
def generate_ERA5_3dCube(varName, xCoordName, yCoordName, tCoordName, beforeNC4File, duringNC4File,
                        afterNC4File, anaDateTime, tVals):
    # extract coordinate values from beforeNC4File, or report error and return None values
    if isfile(beforeNC4File):
        nc4Hdl = Dataset(beforeNC4File)
        try:
            xCoord = np.asarray(nc4Hdl.variables[xCoordName]).squeeze()
            yCoord = np.asarray(nc4Hdl.variables[yCoordName]).squeeze()
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordBefore = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + beforeNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + beforeNC4File + ' does not exist')
        return None, None, None, None, None
    # extract time-coordinate values from duringNC4File, or report error and return None values
    if isfile(duringNC4File):
        nc4Hdl = Dataset(duringNC4File)
        try:
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordDuring = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + duringNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + duringNC4File + ' does not exist')
        return None, None, None, None, None
    # extract time-coordinate values from afterNC4File, or report error and return None values
    if isfile(afterNC4File):
        nc4Hdl = Dataset(afterNC4File)
        try:
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordAfter = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + afterNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + afterNC4File + ' does not exist')
        return None, None, None, None, None
    # define tCoord as tVals
    tCoord = tVals
    # define dimension-sizes
    ny = np.size(yCoord)
    nx = np.size(xCoord)
    nt = np.size(tCoord)
    # generate a NaN 3D-cube of dimension (ny,nx,nt)
    cube3d = np.nan * np.ones((ny,nx,nt))
    # loop through hours of observation-window (-3:+3), pick NC4 file
    for t in range(nt):
        # identify the appropriate ERA5 file, if it doesn't exist its time-level will remain NaN
        # if requesting an unknown time, raise error and loop (current time in loop will be NaN)
        if np.isin(tCoord[t], tCoordBefore):
            nc4File = beforeNC4File
            nc4Hour = np.where(tCoordBefore==tCoord[t])[0][0]
        elif np.isin(tCoord[t], tCoordDuring):
            nc4File = duringNC4File
            nc4Hour = np.where(tCoordDuring==tCoord[t])[0][0]
        elif np.isin(tCoord[t], tCoordAfter):
            nc4File = afterNC4File
            nc4Hour = np.where(tCoordAfter==tCoord[t])[0][0]
        else:
            print('No file available for time T=', t)
        # fill cube3d at appropriate time-level, if nc4File exists and contains varName, otherwise
        # leave time-level NaN
        if isfile(nc4File):
            nc4Hdl = Dataset(nc4File)
            a = np.asarray(nc4Hdl.variables[varName]).squeeze()
            cube3d[:,:,t] = a[nc4Hour,:,:].squeeze()
    # Return coordinate vectors and 3D cube
    return yCoord, xCoord, tCoord, cube3d


# generate_ERA5_4dCube: append netCDF 3D fields across different times into a single 4D cube
# INPUTS:
#   varName: name of variable to extract from ERA5 GRIB files (string)
#   xCoordName: name of coordinate-variable in x-direction in ERA5 NC4 files (string)
#   yCoordName: name of coordinate-variable in y-direction in ERA5 NC4 files (string)
#   zCoordName: name of coordinate-variable in z-direction in ERA5 NC4 files (string)
#   tCoordName: name of coordinate-variable in t-direction in ERA5 NC4 files (string)
#   beforeNC4File: full-path name of ERA5 NC4 file at "before" period relative to analysis time-window (string)
#   duringNC4File: full-path name of ERA5 NC4 file at "during" period relative to analysis time-window (string)
#   afterNC4File: full-path name of ERA5 NC4 file at "after" period relative to analysis time-window (string)
#   anaDateTime: date-time of analysis time (datetime.datetime() object)
#   tVals: vector of values along tCoord to extract into 4dCube (float vector)
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
# datetime
# os
def generate_ERA5_4dCube(varName, xCoordName, yCoordName, zCoordName, tCoordName, beforeNC4File, duringNC4File,
                        afterNC4File, anaDateTime, tVals):
    # extract coordinate values from beforeNC4File, or report error and return None values
    if isfile(beforeNC4File):
        nc4Hdl = Dataset(beforeNC4File)
        try:
            xCoord = np.asarray(nc4Hdl.variables[xCoordName]).squeeze()
            yCoord = np.asarray(nc4Hdl.variables[yCoordName]).squeeze()
            zCoord = np.asarray(nc4Hdl.variables[zCoordName]).squeeze()
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordBefore = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + beforeNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + beforeNC4File + ' does not exist')
        return None, None, None, None, None
    # extract time-coordinate values from duringNC4File, or report error and return None values
    if isfile(duringNC4File):
        nc4Hdl = Dataset(duringNC4File)
        try:
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordDuring = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + duringNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + duringNC4File + ' does not exist')
        return None, None, None, None, None
    # extract time-coordinate values from afterNC4File, or report error and return None values
    if isfile(afterNC4File):
        nc4Hdl = Dataset(afterNC4File)
        try:
            # tCoord is presumed to be in hours since base dateTime (1900,1,1,0), translate to hours relative
            # to analysis dateTime
            tCoordNative = np.asarray(nc4Hdl.variables[tCoordName]).squeeze()
            baseDateTime = datetime.datetime(1900,1,1,0)
            dateTimes = [baseDateTime + datetime.timedelta(hours=int(t)) for t in tCoordNative]
            tCoordAfter = np.asarray([(dt-anaDateTime).total_seconds()/3600. for dt in dateTimes]).squeeze()
        except:
            print('NC4 FILE: ' + afterNC4File + ' exists, but coordinate variables raise error')
            return None, None, None, None, None
    else:
        print('NC4 FILE: ' + afterNC4File + ' does not exist')
        return None, None, None, None, None
    # define tCoord as tVals
    tCoord = tVals
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
        if np.isin(tCoord[t], tCoordBefore):
            nc4File = beforeNC4File
            nc4Hour = np.where(tCoordBefore==tCoord[t])[0][0]
        elif np.isin(tCoord[t], tCoordDuring):
            nc4File = duringNC4File
            nc4Hour = np.where(tCoordDuring==tCoord[t])[0][0]
        elif np.isin(tCoord[t], tCoordAfter):
            nc4File = afterNC4File
            nc4Hour = np.where(tCoordAfter==tCoord[t])[0][0]
        else:
            print('No file available for time T=', t)
        # fill cube4d at appropriate time-level, if gribFile exists and contains varName, otherwise
        # leave time-level NaN
        if isfile(nc4File):
            nc4Hdl = Dataset(nc4File)
            a = np.asarray(nc4Hdl.variables[varName]).squeeze()
            cube4d[:,:,:,t] = a[nc4Hour,:,:,:].squeeze()
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
#   numDup: number of points to duplicate along perDim dimension on each side
#
# OUTPUTS:
#   outputArray: same as inputArray, but with duplicate points on either side of periodic dimension (numPy array)
#   perCoordNew: same as perCoord, but with fictitious coordinate-values for duplicate points on either side of dimension (numPy array)
#
# DEPENDENCIES:
#   numpy
def duplicate_periodic_dimension_points(inputArray, perDim, perCoord, numDup=1):
    # introduce wrap-padding of numDup to all axes of inputArray
    wrapArray = np.pad(inputArray,pad_width=numDup,mode='wrap')
    # define slicing of wrapArray cut down along all dimension-edges except perDim
    slc = [slice(numDup, -numDup, 1)] * len(wrapArray.shape)  # slice all dimensions from [1:-1:1]
    slc[perDim] = slice(None)  # slice perDim array from [:] instead
    deltaCoord = np.diff(perCoord)[0]  # all np.diff() values, the step between values of periodic coordinate, should be identical
    # create new perCoord with expanded dimension
    perCoordNew = np.nan * np.ones((np.size(perCoord)+2*numDup,))
    perCoordNew[numDup:-numDup] = perCoord
    for j in range(0,numDup):
        perCoordNew[j] = perCoord[0] - deltaCoord*(numDup-j)
    for j in range(0,numDup):
        perCoordNew[numDup+np.size(perCoord)+j] = perCoord[-1] + deltaCoord*(j+1)
    return wrapArray[tuple(slc)], perCoordNew
#
# begin
#
if __name__ == "__main__":
    #
    # take user inputs
    #
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directories and names of ' +
                                                 'netCDF super-ob and ERA5 netCDF files')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    parser.add_argument('anaWindowBeg', metavar='ANAWINDOWBEG', type=str, help='beginning of analysis-window portion to process')
    parser.add_argument('anaWindowEnd', metavar='ANAWINDOWEND', type=str, help='end of analysis-window portion to process')
    parser.add_argument('tileExt', metavar='TILEEXT', type=str, help='extension from center gridpoint to define 2D field-tile')
    parser.add_argument('obDataDir', metavar='OBDATADIR', type=str, help='full path to data directory for super-observations')
    parser.add_argument('obFile', metavar='INFILE', type=str, help='input netCDF superob file')
    parser.add_argument('era5DataDir', metavar='ERA5DATADIR', type=str, help='full path to data directory for ERA5')
    parser.add_argument('era5TBefore', metavar='ERA5TBEFORE', type=str, help='file-name for ERA5 T data from before-period')
    parser.add_argument('era5TDuring', metavar='ERA5TDURING', type=str, help='file-name for ERA5 T data from during-period')
    parser.add_argument('era5TAfter', metavar='ERA5TAFTER', type=str, help='file-name for ERA5 T data from after-period')
    parser.add_argument('era5SPBefore', metavar='ERA5SPBEFORE', type=str, help='file-name for ERA5 SP data from before-period')
    parser.add_argument('era5SPDuring', metavar='ERA5SPDURING', type=str, help='file-name for ERA5 SP data from during-period')
    parser.add_argument('era5SPAfter', metavar='ERA5SPAFTER', type=str, help='file-name for ERA5 SP data from after-period')
    parser.add_argument('outputNetcdfFile', metavar='OUTFILE', type=str, help='output netCDF file (superobs with metadata)')
    # parse arguments
    userInputs = parser.parse_args()
    #userInputs = parser.parse_args([
    #                                 '2023010100',
    #                                 '-3.0',
    #                                 '-2.0',
    #                                 '10',
    #                                 '/Users/bhoover/Desktop/IMSG/PROJECTS/ML_superob',
    #                                 'gdas.t00z.satwnd.tm00.bufr_d_2023010100_superobs.nc_TEST',
    #                                 '/Users/bhoover/Desktop/IMSG/PROJECTS/ML_superob',
    #                                 'e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2022123118_2022123123.nc',
    #                                 'e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2023010100_2023010105.nc',
    #                                 'e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2023010106_2023010111.nc',
    #                                 'e5.oper.an.ml.128_134_sp.regn320sc.2022123118_2022123123.nc',
    #                                 'e5.oper.an.ml.128_134_sp.regn320sc.2023010100_2023010105.nc',
    #                                 'e5.oper.an.ml.128_134_sp.regn320sc.2023010106_2023010111.nc',
    #                                 'gdas.t00z.satwnd.tm00.bufr_d_2023010100_superobs_T_tiles_ERA5_t01.nc'
    #                               ])
    # quality-control inputs:
    # if userInputs.obDataDir does not end in '/', append it
    obDataDir = userInputs.obDataDir + '/' if userInputs.obDataDir[-1] != '/' else userInputs.obDataDir
    # if userInputs.era5DataDir does not end in '/', append it
    era5DataDir = userInputs.era5DataDir + '/' if userInputs.era5DataDir[-1] != '/' else userInputs.era5DataDir
    # define analysis datetime
    anaDateTime = datetime.datetime.strptime(userInputs.anaDateTime, '%Y%m%d%H')
    # define anaWindowBeg and anaWindowEnd as float-values
    anaWindowBeg = float(userInputs.anaWindowBeg)
    anaWindowEnd = float(userInputs.anaWindowEnd)
    # define tileExt as integer
    tileExt = int(userInputs.tileExt)
    # define anaWindowVals
    anaWindowVals = np.asarray([anaWindowBeg, anaWindowEnd])
    #
    # ingest and filter superob data
    #
    t0 = time()
    # load superob data: latitude, longitude, pressure, time
    hdl = Dataset(obDataDir + userInputs.obFile)
    lat = np.asarray(hdl.variables['lat']).squeeze()  # latitude (float, deg)
    lon = np.asarray(hdl.variables['lon']).squeeze()  # longitude (float, deg)
    pre = np.asarray(hdl.variables['pre']).squeeze()  # pressure (float, Pa)
    tim = np.asarray(hdl.variables['tim']).squeeze()  # time (float, frac. hrs relative to anaDateTime)
    # filter observations for those only within anaWindowVals
    idx = np.where((tim>=np.min(anaWindowVals)) & (tim<np.max(anaWindowVals)))[0]  # contains tim [min, max) of anaWindowVals
    lat = lat[idx]
    lon = lon[idx]
    pre = pre[idx]
    tim = tim[idx]
    t1 = time()
    print('collected and filtered superob data in {:.2f} seconds'.format(t1-t0))
    #
    # ingest model dimensions
    #
    t0 = time()
    # pull ERA5 lat, lon, hybrid-sigma parameters from era5TBefore (should be identical for all ERA5 files)
    hdl=Dataset(era5DataDir + userInputs.era5TBefore)
    grdLat = np.asarray(hdl.variables['latitude']).squeeze()   # model grid latitudes (float, deg, descending)
    grdLon = np.asarray(hdl.variables['longitude']).squeeze()  # model grid longitudes (float, deg, 0-360)
    grdAk = np.asarray(hdl.variables['a_model']).squeeze()     # model grid ak values (float, dimensionless)
    grdBk = np.asarray(hdl.variables['b_model']).squeeze()     # model grid bk values (float, dimensionless)
    # define grid dimensions
    nz = hdl.dimensions['level'].size
    ny = hdl.dimensions['latitude'].size
    nx = hdl.dimensions['longitude'].size
    t1 = time()
    print('collected model grid information in {:.2f} seconds'.format(t1-t0))
    #
    # ingest model field (T) into 4D-cube
    #
    t0 = time()
    # generate 4dcube of temperature data within tVals
    cubeOut = era5_4dcube_T = generate_ERA5_4dCube(varName="T",
                                                   xCoordName="longitude",
                                                   yCoordName="latitude",
                                                   zCoordName="level",
                                                   tCoordName="time",
                                                   beforeNC4File=era5DataDir + userInputs.era5TBefore,
                                                   duringNC4File=era5DataDir + userInputs.era5TDuring,
                                                   afterNC4File=era5DataDir + userInputs.era5TAfter,
                                                   anaDateTime=anaDateTime,
                                                   tVals=anaWindowVals)
    # assign cubeOut tuple-values to variables
    zCoord = cubeOut[0]
    yCoord = cubeOut[1]
    xCoord = cubeOut[2]
    tCoord = cubeOut[3]
    cube4d_T = cubeOut[4]
    t1 = time()
    print('generated 4d model T data cube in {:.2f} seconds'.format(t1-t0))
    #
    # ingest model SP field into 3D-cube
    #
    t0 = time()
    # generate 3Dcube of surface pressure data within tVals
    cubeOut = era5_3dcube_PS = generate_ERA5_3dCube(varName="SP",
                                                    xCoordName="longitude",
                                                    yCoordName="latitude",
                                                    tCoordName="time",
                                                    beforeNC4File=era5DataDir + userInputs.era5SPBefore,
                                                    duringNC4File=era5DataDir + userInputs.era5SPDuring,
                                                    afterNC4File=era5DataDir + userInputs.era5SPAfter,
                                                    anaDateTime=anaDateTime,
                                                    tVals=anaWindowVals)
    # we only need the 3Dcube from this run, skip the coordinate values
    cube3d_SP = cubeOut[3]
    t1 = time()
    print('generated 3d model SP data cube in {:.2f} seconds'.format(t1-t0))
    #
    # extend model data-cubes in longitude dimension by tileExt+1
    #
    t0 = time()
    # make duplicative columns on edges of grids along longitude dimension
    cube3d_SP_dup, xCoord_dup = duplicate_periodic_dimension_points(cube3d_SP, 1, xCoord, numDup=tileExt+1)
    cube4d_T_dup, xCoord_dup = duplicate_periodic_dimension_points(cube4d_T, 2, xCoord, numDup=tileExt+1)
    # grdLat is north-to-south convention, reverse this for coordinate and data
    cube3d_SP_dup = np.flip(cube3d_SP_dup, axis=0)
    cube4d_T_dup = np.flip(cube4d_T_dup, axis=1)
    yCoord = np.flip(yCoord, axis=0)
    t1 = time()
    print('extended model data cubes in {:.2f} seconds'.format(t1-t0))
    #
    # compute model sigma values at each point on extended grid
    #
    t0 = time()
    # compute sigma-value of each model-level as ak/p0 + bk
    cube4d_Sig_dup = np.nan * np.ones((len(zCoord),len(yCoord),len(xCoord_dup),len(tCoord)))
    for t in range(len(tCoord)):
        for k in range(len(zCoord)):
            cube4d_Sig_dup[k,:,:,t] = grdBk[k] + grdAk[k] * cube3d_SP_dup[:,:,t].squeeze()**-1.
    t1 = time()
    print('generated (extended) 4d model data sigma cube in {:.2f} seconds'.format(t1-t0))
    #
    # interpolate model surface pressure to superob (lat,lon,tim) and compute model sigma-level of superobs
    #
    t0 = time()
    # interpolate the grdPS values to ob locations and times
    dims3dCube = (yCoord, xCoord_dup, tCoord)  # set of 2d-plane dimensions for linear interpolation
    lonFix = lon.copy()
    fix = np.where(lonFix<0.)
    lonFix[fix] = lonFix[fix] + 360.
    Psbk = interpn(dims3dCube, cube3d_SP_dup, np.asarray([lat, lonFix, tim]).T)
    # map ob pressure values to model sigma values
    sigPre = grdAk + Psbk[:,None]*grdBk[None,:]
    sigVal = grdAk[None,:]/Psbk[:,None] + grdBk
    # perform 1D interpolation to find sigma-value at ob pressure value for each ob
    sig = np.asarray([np.interp(pre[i],sigPre[i,:], sigVal[i,:]) for i in range(len(pre))]).squeeze()
    t1 = time()
    print('generated model-sigma ob vectors in {:.2f} seconds'.format(t1-t0))
    #
    # compute 2D model T tiles centered on each superob
    #
    allIdx = np.arange(np.size(lat))         # all obs
    yesIdx = np.where(np.abs(lat)<86.5)[0]   # obs within +/- 86.5 deg latitude (more poleward == not enough data for full tile)
    noIdx = np.setdiff1d(allIdx, yesIdx)     # obs poleward of +/- 86.5 deg latitude
    obIdx = list(allIdx)                     # obs selected for processing among all obs
    yi = list(np.intersect1d(obIdx,yesIdx))  # selected obs within +/- 86.5 deg latitude
    ni = list(np.intersect1d(obIdx,noIdx))   # selected obs poleward of +/- 86.5 deg latitude
    # create a 3D tile array with a time-dimension to store 2D tiles at each time-level of anaWindowVals
    grdT_tile3D = np.zeros((np.size(obIdx), 2*tileExt+1, 2*tileExt+1, np.size(tCoord)))
    # create model lat/lon meshgrids on extended-map
    grdLonMesh, grdLatMesh = np.meshgrid(xCoord_dup,yCoord)
    # for each time-level, construct a tile at the nearest model-level centered on the observation
    t0 = time()
    Nji = np.asarray([np.unravel_index(np.argmin((grdLatMesh-lat[i])**2. + (grdLonMesh-lonFix[i])**2.),
                                                  shape=np.shape(grdLatMesh)) for i in obIdx])
    t1 = time()
    print('horizontal nearest neighbors found in {:.2f} seconds'.format(t1-t0))
    for t in range(np.size(tCoord)):
        t0 = time()
        Nk = np.asarray([np.argmin((cube4d_Sig_dup[:,Nji[obIdx.index(i)][0],Nji[obIdx.index(i)][1],t].squeeze()-sig[i])**2.) for i in obIdx])
        Njiy = Nji[np.where(np.isin(obIdx,yi)),:].squeeze()
        Nky = Nk[np.where(np.isin(obIdx,yi))].squeeze()
        grdT_tile3D[np.where(np.isin(obIdx,yi)),:,:,t] = np.asarray([cube4d_T_dup[Nky[yi.index(i)],Njiy[yi.index(i)][0]-tileExt:Njiy[yi.index(i)][0]+tileExt+1,Njiy[yi.index(i)][1]-tileExt:Njiy[yi.index(i)][1]+tileExt+1,t].squeeze() for i in yi])
        t1 = time()
        print('processing tiles on time-level {:d} completed in {:.2f} seconds'.format(t,t1-t0))
    t0 = time()
        # define time-weighting coefficients manually:
    #
    # t1 |-----------delt1----------|t|-----delt2------|t2
    #    |---------------------delt--------------------|
    #
    # delt = delt1 + delt2
    # w1 = delt-delt1/delt = delt2/delt
    # w2 = delt-delt2/delt = delt1/delt
    delt1 = tim[obIdx] - tCoord[0]
    delt2 = tCoord[1] - tim[obIdx]
    delt = tCoord[1] - tCoord[0]
    w1 = delt2/delt
    w2 = delt1/delt
    # construct a single 2D tile for each observation that is linearly interpolated between time-levels
    grdT_tile2D = w1[:,None,None] * grdT_tile3D[:,:,:,0] + w2[:,None,None] * grdT_tile3D[:,:,:,1]
    t1 = time()
    print('generated 2d model T tiles in {:.2f} seconds'.format(t1-t0))
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
    y  = ncOut.createDimension( 
                                 'y' , # nc_out.createDimension input: Dimension name 
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    x  = ncOut.createDimension( 
                                 'x' , # nc_out.createDimension input: Dimension name 
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    # add variables
    T = ncOut.createVariable(
                                  'T'       ,
                                  'f8'        ,
                                  ('ob', 'y', 'x')
                            )
    xi = ncOut.createVariable(
                                  'idx'     ,
                                  'i8'        ,
                                  ('ob')
                             )
    latVec = ncOut.createVariable(
                                   'lat'     ,
                                   'f8'        ,
                                   ('ob', 'y')
                                 )
    lonVec = ncOut.createVariable(
                                   'lon'     ,
                                   'f8'        ,
                                   ('ob', 'x')
                                 )
    # assign netCDF file attributes
    ncOut.labelSource = ("ERA5 hourly data on native model levels 1940 to present, reanalysis temperature and surface pressure on 0.25 " + 
                         "degree global grids on 128 model levels, via NCAR RDA " + 
                         "and interpolated linearly in latitude, longitude, and time")
    ncOut.labelCitation = ("European Centre for Medium-Range Weather Forecasts. 2022, updated . ERA5 Reanalysis Model Level Data. " +
                           "Research Data Archive at the National Center for Atmospheric Research, Computational and Information " + 
                           "Systems Laboratory. https://doi.org/10.5065/XV5R-5344")
    ncOut.analysisDateTime = userInputs.anaDateTime
    ncOut.sourceFile = userInputs.obFile
    # fill netCDF file variables with labels
    T[:,:,:]      = grdT_tile2D  # 2D tile of field
    xi[:]         = idx[obIdx]  # index of observations in full superob file
    # provide filler dummy values to lonVec and latVec for all obs in ni, real values for all obs in yi
    latyi = np.asarray([yCoord[Nji[i][0]-tileExt:Nji[i][0]+tileExt+1] for i in yi]).squeeze()
    latni = np.asarray([np.nan*np.ones((2*tileExt+1)) for i in ni]).squeeze()
    lonyi = np.asarray([xCoord_dup[Nji[i][1]-tileExt:Nji[i][1]+tileExt+1] for i in yi]).squeeze()
    lonni = np.asarray([np.nan*np.ones((2*tileExt+1)) for i in ni]).squeeze()
    # generate whole *Vec as zero-array and fill with *yi and *ni values
    latV = np.zeros((len(yi)+len(ni),2*tileExt+1))
    lonV = np.zeros((len(yi)+len(ni),2*tileExt+1))
    latV[yi,:] = latyi
    latV[ni,:] = latni
    lonV[yi,:] = lonyi
    lonV[ni,:] = lonni
    # fill latVec and lonVec
    latVec[:,:] = latV  # latitude along y-dimension of tile
    lonVec[:,:] = lonV  # longitude along x-dimension of tile
    # close ncOut
    ncOut.close()
#
# end
#
