import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pyproj
import argparse
# BTH NOTE: The following imports are temporarily diverted to internal functions, but should be
#           returned to imports for the live code
# from bulk_stats_dependencies import spddir_to_uwdvwd
# from bulk_stats_dependencies import uwdvwd_to_spddir
def spddir_to_uwdvwd(spd,ang):
    uwd=-spd*np.sin(ang*(np.pi/180.))
    vwd=-spd*np.cos(ang*(np.pi/180.))
    return uwd, vwd

def uwdvwd_to_spddir(uwd,vwd):
    spd=np.sqrt(uwd**2.+vwd**2.)
    ang=(270.-np.arctan2(vwd,uwd)*(180./np.pi))%(360.)
    return spd, ang
#
# define internal functions
#
# array_int_to_str: given an array of integers, return a concatenated string of the integer values
#
# INPUTS:
#   a: array of input integeger values (int array)
#
# OUTPUTS:
#   integer-values concatenated together into a string with no delimiters (str)
#
# DEPENDENCIES:
#   none 
def array_int_to_str(a):
    return ''.join(map(str, a))


#
# define_superobs: compute super-observations and super-ob metadata from observations with assigned clusters
#
# INPUTS:
#   lat: vector of ob latitude (float, deg)
#   lon: vector of ob longitude (float, deg)
#   pre: vector of ob pressure (float, Pa)
#   tim: vector of ob time (float, fractional hrs. relative to analysis datetime)
#   uwd: vector of ob u-wind component (float, m/s)
#   vwd: vector of ob v-wind component (float, m/s)
#   typ: vector of ob type (integer, categorical)
#   cid: vector of ob cluster-ID (integer, categorical)
#   sid: vector of ob satellite-IDs (integer, categorical)
#   qin: vector of ob quality indicator values (integer, unitless)
#   allTypes: vector of all possible ob-types among obs (integer, categorical)
#
# OUTPUTS:
#   SOlat: vector of super-ob latitude (float, deg)
#   SOlon: vector of super-ob longitude (float, deg)
#   SOpre: vector of super-ob pressure (float, Pa)
#   SOtim: vector of super-ob time (float, fractional hrs. relative to analysis datetime)
#   SOuwd: vector of super-ob u-wind component (float, m/s)
#   SOvwd: vector of super-ob v-wind component (float, m/s)
#   SOnob: vector of super-ob number of member observations (integer)
#   SOnty: vector of super-ob number of member ob-types (integer)
#   SOnid: vector of super-ob number of member satellite-IDs (integer)
#   SOqim: vector of super-ob mean quality indicator value (float, unitless)
#   SOuvr: vector of super-ob variance of member u-wind components (float, (m/s)**2)
#   SOvvr: vector of super-ob variance of member v-wind components (float, (m/s)**2)
#   SOpvr: vector of super-ob variance of member pressures (float, Pa**2)
#   SOtvr: vector of super-ob variance of member times (float, (frc. hrs)**2)
#   SOdvr: vector of super-ob variance in [x,y]-space, as variance in distance from epoch point (float, (m)**2.)
#   SOqvr: vector of super-ob variances in quality indicator values (float, unitless**2)
#   SOtyp: vector of (typeValue,)-dimension vectors identifying the presence (=1) or absence (=0) of each ob-type
#          present in file
#
# DEPENDENCIES:
#   numpy
#   pyproj
def define_superobs(lat, lon, pre, tim, uwd, vwd, typ, cid, sid, qin, allTypes):
    # create an internal function to count then number of unique typ values for an array of vectors
    def num_types(a):
        return np.size(np.unique(a))
    # create an internal function to track the presence or absence of each unique typ value in SO
    def track_types(a):
        numTypes = np.size(allTypes)
        typTracker = np.zeros((numTypes),)
        return np.isin(allTypes, a).astype('int32')
    # create a pyproj.Transformer() object to convert between EPSG:4087 (x,y)-space and EPSG:4326 (lon,lat)-space
    proj_xy_to_ll = pyproj.Transformer.from_crs(4087, 4326, always_xy=True)
    proj_ll_to_xy = pyproj.Transformer.from_crs(4326, 4087, always_xy=True)
    # define an epoch point at lat=0., lon=0. for reference
    epochX, epochY = proj_ll_to_xy.transform(0., 0.)
    # define full list of cluster-IDs and their counts
    uniqueCIDs, cidCounts = np.unique(cid, return_counts=True)
    nCIDs = np.size(uniqueCIDs)
    # define a dictionary that translates each unique CID to a corresponding index-value
    CIDdict={}
    for i in range(np.size(uniqueCIDs)):
        CIDdict[uniqueCIDs[i]]=i
    # define size of vector of all ob-types
    numTypes = np.size(allTypes)
    # define superOb (meta)data arrays
    SOlat = np.nan*np.ones((nCIDs,))  # latitude (float, deg)
    SOlon = np.nan*np.ones((nCIDs,))  # longitude (float, deg)
    SOpre = np.nan*np.ones((nCIDs,))  # pressure (float, Pa)
    SOtim = np.nan*np.ones((nCIDs,))  # time (float, fractional hrs)
    SOuwd = np.nan*np.ones((nCIDs,))  # u-wind (float, m/s)
    SOvwd = np.nan*np.ones((nCIDs,))  # v-wind (float, m/s)
    SOnob = np.nan*np.ones((nCIDs,))  # number of AMV members (integer)
    SOnty = np.nan*np.ones((nCIDs,))  # number of AMV types among members (integer)
    SOnid = np.nan*np.ones((nCIDs,))  # number of AMV satellite-IDs among members (integer)
    SOqim = np.nan*np.ones((nCIDs,))  # mean of AMV quality indicator values (float, unitless)
    SOuvr = np.nan*np.ones((nCIDs,))  # u-wind variance among members (float, m^2/s^2)
    SOvvr = np.nan*np.ones((nCIDs,))  # v-wind variance among members (float, m^2/s^2)
    SOpvr = np.nan*np.ones((nCIDs,))  # pressure variance among members (float, Pa^2)
    SOtvr = np.nan*np.ones((nCIDs,))  # time variance among members (float, frac. hrs.^2)
    SOdvr = np.nan*np.ones((nCIDs,))  # [x,y]-space variance among members as distance from epoch point (float, m^2)
    SOqvr = np.nan*np.ones((nCIDs,))  # QI variance among members (float, unitless^2)
    SOtyp = np.nan*np.ones((nCIDs, numTypes)) # each ob-type in allTypes: present (=1) or absent (=0) from super-ob (integer)
    # start with the singleton (1-member) "clusters", and assign directly to the superOb arrays
    n = 1
    c = uniqueCIDs[np.where(cidCounts==n)[0]]  # all CIDs with cidCounts==n
    x = np.where(np.isin(cid, c))[0]  # all AMV indices with CIDs in c
    i = np.asarray([CIDdict.get(key) for key in c])  # corresponding SO* array indices for CIDs
    if np.size(c) > 0:
        # since these are singleton clusters, the ob values and location data can be assigned
        # directly, SOnob and SOnty are both 1., and all SO variances are 0., by default
        SOlat[i] = lat[x]
        SOlon[i] = lon[x]
        SOpre[i] = pre[x]
        SOtim[i] = tim[x]
        SOuwd[i] = uwd[x]
        SOvwd[i] = vwd[x]
        SOnob[i] = 1.
        SOnty[i] = 1.
        SOnid[i] = 1.
        SOqim[i] = qin[x]
        SOuvr[i] = 0.
        SOvvr[i] = 0.
        SOpvr[i] = 0.
        SOtvr[i] = 0.
        SOdvr[i] = 0.
        SOqvr[i] = 0.
        # track_types can be applied to each singleton cluster by reshaping to (-1,1) and
        # applying function to each element along axis=1
        SOtyp[i, :] = np.apply_along_axis(func1d=track_types, axis=1, arr=typ[x].reshape(-1,1))
    # loop through the rest of the potential values of n from 2 to max(cidCounts)
    for n in np.arange(start=2, stop=max(cidCounts)+1, step=1):
        c = uniqueCIDs[np.where(cidCounts==n)[0]]  # all CIDs with cidCounts==n
        x = np.where(np.isin(cid, c))[0]  # all AMV indices with CIDs in c
        i = np.asarray([CIDdict.get(key) for key in c])  # corresponding SO* array indices for CIDs
        if np.size(c) > 0:
            # transform indices from x into an [nCID, n] array with unique CID values as the
            # 0-axis and n indices for each cluster as the 1-axis
            #
            # this is achieved by looping through n times and pulling out an array of index
            # values from x that composes a list of the unique CIDs
            ux, ix = np.unique(cid[x],return_index=True)
            SOIdx = -1 * np.ones((np.size(ux),n)).astype('int')
            for j in range(n):
                ux, ix = np.unique(cid[x],return_index=True)
                SOIdx[:,j] = x[ix]
                x = np.setdiff1d(x,x[ix])
            # since these are all non-singleton clusters, we have to compute some values:
            #  1. SOlat and SOlon have to be computed by projecting the AMV lat/lon values into
            #     EPSG:4087 x/y space, averaged, and then projected back to EPSG:4326 lat/lon space
            memXYs = proj_ll_to_xy.transform(lon[SOIdx], lat[SOIdx])
            meanXs = np.mean(memXYs[0], axis=1)
            meanYs = np.mean(memXYs[1], axis=1)
            meanLons, meanLats = proj_xy_to_ll.transform(meanXs, meanYs)
            #     compute x/y space distance of each member from epoch point lat=0., lon=0.
            distXYs = np.sqrt((memXYs[0] - epochX)**2. + (memXYs[1] - epochY)**2.)
            meanLons, meanLats = proj_xy_to_ll.transform(meanXs, meanYs)
            SOlat[i] = meanLats
            SOlon[i] = meanLons
            #  2. SO{pre,tim,uwd,vwd} can all be calculated as a mean value along the 1-axis
            #     instead of a straightforward mean of pre, we will instead perform a mean
            #     of log10(pre) and compute SOpre as np.power(10.,np.mean(np.log10(pre[SOIdx]),axis=1))
            SOpre[i] = np.power(10., np.mean(np.log10(pre[SOIdx]), axis=1))
            SOtim[i] = np.mean(tim[SOIdx], axis=1)
            #     we will add a resizing of the (SOuwd,SOvwd) vector to the mean wind-speed
            #     among AMV members, to account for cancelation of u/v components in
            #     averaging
            #
            #     we start by computing the base superob u- and v-components as a mean of the
            #     AMV members' u- and v-components
            SOuwd[i] = np.mean(uwd[SOIdx], axis=1)
            SOvwd[i] = np.mean(vwd[SOIdx], axis=1)
            #     the mean speed of AMVs in the superob is computed, along with the
            #     superob direction and (current) speed
            AMVSpeedMean = np.mean(np.sqrt(uwd[SOIdx]**2. + vwd[SOIdx]**2.), axis=1)
            SOSpeed, SODirec = uwdvwd_to_spddir(SOuwd[i], SOvwd[i])
            #     the delta between the mean AMV speed and superob speed is computed (positive-definite quantity)
            AMVSpeedDelta = AMVSpeedMean - SOSpeed
            #     the corresponding u-delta and v-delta components are computed as pointing in the direction
            #     of the superob vector with length equal to the speed-delta
            uDelta, vDelta = spddir_to_uwdvwd(AMVSpeedDelta, SODirec)
            #     the delta values are added to the superob u- and v-components to rescale the superOb vector
            #     to match the mean speed of AMVs in the superob
            SOuwd[i] = SOuwd[i] + uDelta
            SOvwd[i] = SOvwd[i] + vDelta
            #  3. SOnob is equal to n by definition
            SOnob[i] = n
            #  4. SOnty is computed by applying num_types along the 1-axis of typ[SOIdx]
            SOnty[i] = np.apply_along_axis(func1d=num_types, axis=1, arr=typ[SOIdx])
            #  5. SOnid is computed by applying num_types along the 1-axis of sid[SOIdx]
            SOnid[i] = np.apply_along_axis(func1d=num_types, axis=1, arr=sid[SOIdx])
            #  6. SOqim is computed as the mean along the 1-axis of float(qin[SOIdx])
            SOqim[i] = np.mean(qin[SOIdx].astype('float32'), axis=1)
            #  7. SO{uvr,vvr,pvr,tvr,qvr} is calculated as a variance along the 1-axis
            SOuvr[i] = np.var(uwd[SOIdx], axis=1)
            SOvvr[i] = np.var(vwd[SOIdx], axis=1)
            SOpvr[i] = np.var(pre[SOIdx], axis=1)
            SOtvr[i] = np.var(tim[SOIdx], axis=1)
            SOdvr[i] = np.var(distXYs, axis=1)
            SOqvr[i] = np.var(qin[SOIdx].astype('float32'), axis=1)
            # 8. SOtyp is computed by applying track_types along the 1-axis of typ[SOIdx]
            SOtyp[i,:] = np.apply_along_axis(func1d=track_types, axis=1, arr=typ[SOIdx])
    # return SO vectors
    return (SOlat, SOlon, SOpre, SOtim, SOuwd, SOvwd, SOnob, SOnty, SOnid, SOqim, SOuvr, SOvvr, SOpvr, SOtvr, SOdvr, SOqvr, SOtyp)
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directory and name of ' +
                                                 'netCDF AMV/cluster file')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full path to data directory')
    parser.add_argument('inputNetcdfFile', metavar='INFILE', type=str, help='input netCDF file (AMVs with reconciled cluster-IDs)')
    parser.add_argument('outputNetcdfFile', metavar='OUTFILE', type=str, help='output netCDF file (superobs with metadata)')
    # parse arguments
    userInputs = parser.parse_args()
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    # open netCDF file-handle for input AMV data and extract arrays
    hdl = Dataset(dataDir + userInputs.inputNetcdfFile)
    amvLat = np.asarray(hdl.variables['lat']).squeeze()  # AMV latitude (float, deg)
    amvLon = np.asarray(hdl.variables['lon']).squeeze()  # AMV longitude (float, deg)
    amvPre = np.asarray(hdl.variables['pre']).squeeze()  # AMV pressure (float, Pa)
    amvTim = np.asarray(hdl.variables['tim']).squeeze()  # AMV time (float, fractional hrs relative to analysis datetime)
    amvUwd = np.asarray(hdl.variables['uwd']).squeeze()  # AMV u-wind component (float, m/s)
    amvVwd = np.asarray(hdl.variables['vwd']).squeeze()  # AMV v-wind component (float, m/s)
    amvTyp = np.asarray(hdl.variables['typ']).squeeze()  # AMV type (integer, categorical)
    amvSID = np.asarray(hdl.variables['sid']).squeeze()  # AMV satellite-ID (integer, categorical)
    amvQIN = np.asarray(hdl.variables['qin']).squeeze()  # AMV type (integer, unitless)
    amvCID = np.asarray(hdl.variables['cid']).squeeze()  # AMV cluster-ID (integer, categorical)
    # define list of all unique observation-types among AMVs in dataset
    allTypes = np.unique(amvTyp.astype('int32'))
    # compute superobs and metadata
    superobData = define_superobs(amvLat, amvLon, amvPre, amvTim, amvUwd, amvVwd, amvTyp, amvCID, amvSID, amvQIN, allTypes)
    supLat = superobData[0]   # superob latitude (float, deg)
    supLon = superobData[1]   # superob longitude (float, deg)
    supPre = superobData[2]   # superob pressure (float, Pa)
    supTim = superobData[3]   # superob time (float, fractional hrs relative to analysis datetime)
    supUwd = superobData[4]   # superob u-wind component (float, m/s)
    supVwd = superobData[5]   # superob v-wind component (float, m/s)
    supNob = superobData[6]   # superob metadata: number of AMVs in cluster (integer)
    supNty = superobData[7]   # superob metadata: number of AMV types in cluster (integer)
    supNid = superobData[8]   # superob metadata: number of satellite-IDs in cluster (integer)
    supQim = superobData[9]   # superob metadata: mean quality indicator in cluster (float, unitless)
    supUvr = superobData[10]  # superob metadata: variance in u-wind among AMVs in cluster (float, (m/s)**2.)
    supVvr = superobData[11]  # superob metadata: variance in v-wind among AMVs in cluster (float, (m/s)**2.)
    supPvr = superobData[12]  # superob metadata: variance in pressure among AMVs in cluster (float, (Pa)**2.)
    supTvr = superobData[13]  # superob metadata: variance in time among AMVs in cluster (float, (frc. hrs)**2.)
    supDvr = superobData[14]  # superob metadata: variance in [x,y]-space among AMVs in cluster, as distance from epoch point (float, (m)**2.)
    supQvr = superobData[15]  # superob metadata: variance in quality indicator among AMVs in cluster (float, unitless**2.)
    supTyp = superobData[16]  # superob metadata: among types in allTypes, present (=1) or absent (=0) in superob (integer)
    # write superob data to output netCDF file
    ncOutFileName = dataDir + userInputs.outputNetcdfFile
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
    typeValue = ncOut.createDimension(
                                      'typeValue' , # nc_out.createDimension input: Dimension name 
                                       None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                     )
    # add variables
    lat = ncOut.createVariable(
                                  'lat'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    lon= ncOut.createVariable(
                                  'lon'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    pre = ncOut.createVariable(
                                  'pre'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    tim = ncOut.createVariable(
                                  'tim'       ,
                                  'f8'        ,
                                  ('ob')
                                )
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
    nob = ncOut.createVariable(
                                  'nob'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    nty = ncOut.createVariable(
                                  'nty'       ,
                                  'i8'        ,
                                  ('ob')
                               )
    nid = ncOut.createVariable(
                                  'nid'       ,
                                  'i8'        ,
                                  ('ob')
                               )
    qim = ncOut.createVariable(
                                  'qim'       ,
                                  'f8'        ,
                                  ('ob')
                               )
    uvr = ncOut.createVariable(
                                  'uvr'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    vvr = ncOut.createVariable(   'vvr'       ,
                                  'f8'        ,
                                  ('ob')
                              )
    pvr = ncOut.createVariable(
                                  'pvr'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    tvr = ncOut.createVariable(
                                  'tvr'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    dvr = ncOut.createVariable(
                                  'dvr'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    qvr = ncOut.createVariable(
                                  'qvr'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    typ = ncOut.createVariable(
                                  'typ'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    typeValues = ncOut.createVariable(
                                      'typeValues',
                                      'i8'        ,
                                      ('typeValue')
                                     )
    # fill netCDF output file
    lat[:]      = supLat
    lon[:]      = supLon
    pre[:]      = supPre
    tim[:]      = supTim
    uwd[:]      = supUwd
    vwd[:]      = supVwd
    nob[:]      = supNob
    nty[:]      = supNty
    nid[:]      = supNid
    qim[:]      = supQim
    uvr[:]      = supUvr
    vvr[:]      = supVvr
    pvr[:]      = supPvr
    tvr[:]      = supTvr
    dvr[:]      = supDvr
    qvr[:]      = supQvr
    # for typ, convert array of values (as integer) to a concatenated string
    supTypStr = np.apply_along_axis(func1d=array_int_to_str, axis=1, arr=supTyp.astype('int32'))
    # then translate each string of 0's and 1's into a base-2 (binary) integer for output
    f = lambda x: int(x, 2)  # lambda function for converting string to base-2 integer
    # NOTE: the resulting binary-encoded value can be reconstructed into the equivalent string
    #       of 0s and 1s using the following formula:
    #
    #       q = lambda x: format(x, 'b').rjust(np.size(typeValues),'0')
    #
    #       translating back and forth from string_ to binary_value:
    #
    #       binary_value = f(string_value)
    #       string_value = q(binary_value)
    typ[:]    = np.asarray([f(x) for x in supTypStr])
    typeValues[:] = allTypes
    # close ncOut
    ncOut.close()
#
# end
#
