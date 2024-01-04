# ML-labeling compliant
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import bz2
import pickle
import _pickle as cPickle
from glob import glob
import argparse
#
# define internal functions
#
# binary_decode_type: given a vector of binary-encoded "type" integers and a vector of type-values
#                     in the file, return a vector of strings containing "1"s in each place where
#                     the corresponding type-value is present in a superob and "0"s in each place
#                     where the corresponding type-value is absent in a superob.
#
# INPUTS:
#   typ: vector of binary-encoded "type" values (integers, binary-encoded)
#   typeValues: vector of all AMV type-values that are present in the file (integer, categorical)
#
# OUTPUTS:
#   array of strings of same length as typeValues, containing "1" for each present type-value and
#   "0" for each absent type-value in a superob
#
# DEPENDENCIES:
#   numpy
def binary_decode_type(typ, typeValues):
    q = lambda x: format(x, 'b').rjust(np.size(typeValues),'0')
    return np.asarray([q(x) for x in typ]).squeeze()


# search_types: scan binary-decoded "type" values for "1"s and fill a vector of all possible AMV
#               types with 1s for the corresponding typeValues, 0 otherwise
#
# INPUTS:
#   s: vector of binary-decoded "type" strings (string)
#   typeValues: vector of all AMV type-values that are present in the file (integer, categorical)
#   allTypes: vector of all POSSIBLE AMV type-values (integer, categorical)
#
# OUTPUTS:
#   types: vector of length of allTypes with 1 for each present type-value and 0 for all others, per superob
#
# DEPENDENCIES:
#   numpy
def search_types(s, typeValues, allTypes):
    types = np.zeros((np.size(s),np.size(allTypes)))
    for j in range(np.size(typeValues)):
        tv = typeValues[j]
        idx = np.where(allTypes==tv)[0]
        jdx = np.where(np.asarray([int(x[j]) for x in s]).squeeze()==1)[0]
        types[jdx,idx] = 1
    return types


# extract_superob_data: extract data-arrays from input netCDF4 file-handle and return ob-data
#
# INPUTS:
#   hdl: netCDF4.Dataset() file-handle of superob file
#   allTypes: vector of all POSSIBLE AMV types (integer, categorical)
#
# OUTPUTS:
#   tuple containing:
#      lat: superob latitudes (float, deg)
#      lon: superob longitudes (float, deg)
#      pre: superob pressure (float, Pa)
#      tim: superob time (float, frac. hrs from analysis time)
#      uwd: superob u-wind (float, m/s)
#      vwd: superob v-wind (float, m/s)
#      nob: superob number of AMVs (integer)
#      nty: superob number of AMV types (integer)
#      uvr: superob variance of AMV u-wind (m2/s2)
#      vvr: superob variance of AMV v-wind (m2/s2)
#      pvr: superob variance of AMV pressure (Pa2)
#      tvr: superob variance in AMV time (hr2)
#      dvr: superob variance of AMV distances from lat/lon origin (m2)
#      typVec: superob vector of 1s for all present AMV types from allTypes, 0s otherwise (integer, categorical)
#
# DEPENDENCIES
#   numpy
def extract_superob_data(hdl, allTypes):
    typeValues = np.asarray(hdl.variables['typeValues']).squeeze()  # all AMV types in file
    lat = np.asarray(hdl.variables['lat']).squeeze()  # latitude (deg)
    lon = np.asarray(hdl.variables['lon']).squeeze()  # longitude (deg)
    pre = np.asarray(hdl.variables['pre']).squeeze()  # pressure (Pa)
    tim = np.asarray(hdl.variables['tim']).squeeze()  # time (frac. hrs in time-window)
    uwd = np.asarray(hdl.variables['uwd']).squeeze()  # zonal wind component (m/s)
    vwd = np.asarray(hdl.variables['vwd']).squeeze()  # merid wind component (m/s)
    nob = np.asarray(hdl.variables['nob']).squeeze()  # number of AMVs in superob
    nty = np.asarray(hdl.variables['nty']).squeeze()  # number of AMV types in superob
    uvr = np.asarray(hdl.variables['uvr']).squeeze()  # variance in u-component in superob (m2/s2)
    vvr = np.asarray(hdl.variables['vvr']).squeeze()  # variance in v-component in superob (m2/s2)
    pvr = np.asarray(hdl.variables['pvr']).squeeze()  # variance in pressure in superob (Pa2)
    tvr = np.asarray(hdl.variables['tvr']).squeeze()  # variance in time in superob (frac. hrs2)
    dvr = np.asarray(hdl.variables['dvr']).squeeze()  # variance in distance btwn AMVs in superob (m2)
    typ = np.asarray(hdl.variables['typ']).squeeze()  # binary-encoded integer type value
    # process typ
    # perform binary decode of typ into strings of 0s and 1s
    typStr = binary_decode_type(typ, typeValues)
    # perform search to generate vector of 0s and 1s for all known AMV types
    typVec = search_types(typStr, typeValues, allTypes)
    return (lat, lon, pre, tim, uwd, vwd, nob, nty, uvr, vvr, pvr, tvr, dvr, typVec)


# extract_label_data: extract data-arrays from input netCDF4 file-handle and return label-data
#
# INPUTS:
#   hdl: netCDF4.Dataset() file-handle of superob file
#
# OUTPUTS:
#   tuple containing:
#      uwd: label u-wind (float, m/s)
#      vwd: label v-wind (float, m/s)
#
# DEPENDENCIES
#   numpy
def extract_label_data(hdl):
    uwd = np.asarray(hdl.variables['uwd']).squeeze()  # zonal wind label (m/s)
    vwd = np.asarray(hdl.variables['vwd']).squeeze()  # merid wind label (m/s)
    return (uwd, vwd)


# compress_pickle: pickle a file and then compress it into a file with extension
#
# INPUTS:
#   title: file-name WITHOUT .pbz2 file-extension
#   data: data object
#
# OUTPUTS:
#   no explicit output on return, but writes compressed {title}.pbz2 file containing data
#
# DEPENDENCIES:
#   bz2
#   pickle
#   _pickle as cPickle
def compress_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
    return


# decompress_pickle: decompress and extract data object from compressed pickle-file
#
# INPUTS:
#   file: file-name WITH .pbz2 file-extension
#
# OUTPUTS:
#   data: data-object from compressed pickle-file
#
# DEPENDENCIES:
#   bz2
#   pickle
#   _pickle as cPickle
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to subdirectory containing /YYYYMMDDHH subdirectories ' + 
                                                 'containing data, and subdirectory search-string')
    parser.add_argument('dataArchDir', metavar='ARCHDIR', type=str, help='full-path to archive directory containing /YYYYMMDDHH ' +
                        'subdirs')
    parser.add_argument('srchStr', metavar='SRCHSTR', type=str, help='search-string to constrain /YYYYMMDDHH search')
    # parse arguments
    userInputs = parser.parse_args()
    # QC userInputs: if dataArchDir does not end in '/', append it
    dataArchDir = userInputs.dataArchDir + '/' if userInputs.dataArchDir[-1] != '/' else userInputs.dataArchDir
    # find all YYYYMMDDHH subdirectories of dataArchDir by searching on userInputs.srchStr + '*'
    dateList = glob(dataArchDir + userInputs.srchStr + '*')
    dateList.sort()
    # reduce to only the final "YYYYMMDDHH" portion of the directory tree by selecting last component of string delimited by slashes
    for i in range(len(dateList)):
        dateList[i] = dateList[i].split('/')[-1]
    # define all possible ob-types and length of type-dimension
    allTypes = np.arange(240, 261, 1)
    typLen = np.size(allTypes)
    # compute length of obs-dimension
    obsLen = 0
    for date in dateList:
        superobFile = dataArchDir + date + '/gdas.t' + date[-2:] + 'z.satwnd.tm00.bufr_d_' + date + '_superobs.nc'
        hdl = Dataset(superobFile)
        obsLen = obsLen + hdl.dimensions['ob'].size
        hdl.close()
    print('ob-space vector length {:d}'.format(obsLen))
    # initialize data arrays for observations
    lat=np.nan * np.ones((obsLen,))
    lon=np.nan * np.ones((obsLen,))
    pre=np.nan * np.ones((obsLen,))
    tim=np.nan * np.ones((obsLen,))
    uwd=np.nan * np.ones((obsLen,))
    vwd=np.nan * np.ones((obsLen,))
    nob=np.nan * np.ones((obsLen,))
    nty=np.nan * np.ones((obsLen,))
    uvr=np.nan * np.ones((obsLen,))
    vvr=np.nan * np.ones((obsLen,))
    pvr=np.nan * np.ones((obsLen,))
    tvr=np.nan * np.ones((obsLen,))
    dvr=np.nan * np.ones((obsLen,))
    typVec=np.nan * np.ones((obsLen, typLen))
    # initialize data arrays for labels
    ulb=np.nan * np.ones((obsLen,))
    vlb=np.nan * np.ones((obsLen,))
    # fill superob arrays
    idxBeg = 0
    idxEnd = 0
    for date in dateList:
        # define superob file and open file-handle
        superobFile = dataArchDir + date + '/gdas.t' +date[-2:] + 'z.satwnd.tm00.bufr_d_' + date + '_superobs.nc'
        print(superobFile)
        hdl = Dataset(superobFile)
        # define vector length and compute appropriate beginning/ending indices in storage arrays
        vecLen = hdl.dimensions['ob'].size
        idxBeg = idxEnd
        idxEnd = idxBeg + vecLen
        # extract superob data from file
        SOdata = extract_superob_data(hdl, allTypes)
        # fill storage arrays from idxBeg to idxEnd
        lat[idxBeg:idxEnd] = SOdata[0]
        lon[idxBeg:idxEnd] = SOdata[1]
        pre[idxBeg:idxEnd] = SOdata[2]
        tim[idxBeg:idxEnd] = SOdata[3]
        uwd[idxBeg:idxEnd] = SOdata[4]
        vwd[idxBeg:idxEnd] = SOdata[5]
        nob[idxBeg:idxEnd] = SOdata[6]
        nty[idxBeg:idxEnd] = SOdata[7]
        uvr[idxBeg:idxEnd] = SOdata[8]
        vvr[idxBeg:idxEnd] = SOdata[9]
        pvr[idxBeg:idxEnd] = SOdata[10]
        tvr[idxBeg:idxEnd] = SOdata[11]
        dvr[idxBeg:idxEnd] = SOdata[12]
        typVec[idxBeg:idxEnd, :] = SOdata[13]
        # close file-handle
        hdl.close()
    # fill label arrays
    idxBeg = 0
    idxEnd = 0
    for date in dateList:
        # define label file and open file-handle
        labelFile = dataArchDir + date + '/gdas.t' + date[-2:] + 'z.satwnd.tm00.bufr_d_' + date + '_superobs_labels_ERA5.nc'
        print(labelFile)
        hdl = Dataset(labelFile)
        # define vector length and compute appropriate beginning/ending indices in storage arrays
        vecLen = hdl.dimensions['ob'].size
        idxBeg = idxEnd
        idxEnd = idxBeg + vecLen
        # extract label data from file
        LBdata = extract_label_data(hdl)
        # fill storage arrays from idxBeg to idxEnd
        ulb[idxBeg:idxEnd] = LBdata[0]
        vlb[idxBeg:idxEnd] = LBdata[1]
        # close file-handle
        hdl.close()
    # write tuple of training data to compressed pickle file
    compress_pickle('superob_training_data', (lat,lon,pre,tim,uwd,vwd,nob,nty,uvr,vvr,pvr,tvr,dvr,typVec,ulb,vlb))
#
# end
#
