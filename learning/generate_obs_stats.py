# Iterates over superobservation files and collects the mean, standard deviation, minimum value, and maximum value for each predictor
# These statistics are used to normalize predictors for SupErrNet
import numpy as np
from netCDF4 import Dataset
import glob
import argparse
#
# begin
#
if __name__ == "__main__":
    # define argparser with inputs from user
    parser = argparse.ArgumentParser(description='define full-path to observation data subdirectories')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full-path to observation data subdirectories')
    parser.add_argument('outFileName', metavar='OUTFILENAME', type=str, help='full-path to statistics output file to be generated')
    #userInputs = parser.parse_args()
    userInputs = parser.parse_args(['/Users/bhoover/Desktop/IMSG/PROJECTS/ML_superob/conv2d/training',
                                    '/Users/bhoover/Desktop/IMSG/PROJECTS/ML_superob/conv2d/training/superob_stats.nc'])
    # quality-control: if dataDir does not end in '/', add it
    dataDir = userInputs.dataDir if userInputs.dataDir[-1]=='/' else userInputs.dataDir + '/'
    # initialize mean, stdv, minval, and maxval variables for each nonbinary input variable
    # store as a 4-member array: [avgVal,stdVal,minVal,maxVal]
    latStats = np.zeros((4,))
    lonStats = np.zeros((4,))
    preStats = np.zeros((4,))
    timStats = np.zeros((4,))
    uwdStats = np.zeros((4,))
    vwdStats = np.zeros((4,))
    nobStats = np.zeros((4,))
    ntyStats = np.zeros((4,))
    nidStats = np.zeros((4,))
    qimStats = np.zeros((4,))
    uvrStats = np.zeros((4,))
    vvrStats = np.zeros((4,))
    pvrStats = np.zeros((4,))
    tvrStats = np.zeros((4,))
    dvrStats = np.zeros((4,))
    qvrStats = np.zeros((4,))
    # define list of input files
    fileList = glob.glob(dataDir + '*/*_superobs.nc')
    fileList.sort()
    print('found {:d} observation files to process'.format(len(fileList)))
    # initialize ob-count as zero
    nObs = 0
    # iterate through fileList
    for fileName in fileList:
        print('processing file: ' + fileName)
        # open file-handle
        hdl = Dataset(fileName)
        # collect observations
        lat = np.asarray(hdl.variables['lat']).squeeze()
        lon = np.asarray(hdl.variables['lon']).squeeze()
        pre = np.asarray(hdl.variables['pre']).squeeze()
        tim = np.asarray(hdl.variables['tim']).squeeze()
        uwd = np.asarray(hdl.variables['uwd']).squeeze()
        vwd = np.asarray(hdl.variables['vwd']).squeeze()
        nob = np.asarray(hdl.variables['nob']).squeeze()
        nty = np.asarray(hdl.variables['nty']).squeeze()
        nid = np.asarray(hdl.variables['nid']).squeeze()
        qim = np.asarray(hdl.variables['qim']).squeeze()
        uvr = np.asarray(hdl.variables['uvr']).squeeze()
        vvr = np.asarray(hdl.variables['vvr']).squeeze()
        pvr = np.asarray(hdl.variables['pvr']).squeeze()
        tvr = np.asarray(hdl.variables['tvr']).squeeze()
        dvr = np.asarray(hdl.variables['dvr']).squeeze()
        qvr = np.asarray(hdl.variables['qvr']).squeeze()
        hdl.close()
        # accumulate values along first *Stats[0] dimension for computing mean
        k = 0
        latStats[k] = latStats[k] + np.sum(lat)
        lonStats[k] = lonStats[k] + np.sum(lon)
        preStats[k] = preStats[k] + np.sum(pre)
        timStats[k] = timStats[k] + np.sum(tim)
        uwdStats[k] = uwdStats[k] + np.sum(uwd)
        vwdStats[k] = vwdStats[k] + np.sum(vwd)
        nobStats[k] = nobStats[k] + np.sum(nob)
        ntyStats[k] = ntyStats[k] + np.sum(nty)
        nidStats[k] = nidStats[k] + np.sum(nid)
        qimStats[k] = qimStats[k] + np.sum(qim)
        uvrStats[k] = uvrStats[k] + np.sum(uvr)
        vvrStats[k] = vvrStats[k] + np.sum(vvr)
        pvrStats[k] = pvrStats[k] + np.sum(pvr)
        tvrStats[k] = tvrStats[k] + np.sum(tvr)
        dvrStats[k] = dvrStats[k] + np.sum(dvr)
        qvrStats[k] = qvrStats[k] + np.sum(qvr)
        # update minVal and maxVal: if nObs == 0, this is the first exposure to observations
        #                           and we set them directly from the data, otherwise we
        #                           compare to existing min/maxVal for update
        # minVal (*Stats[2])
        k = 2
        latStats[k] = np.min([latStats[k],np.min(lat)]) if nObs > 0 else np.min(lat)
        lonStats[k] = np.min([lonStats[k],np.min(lon)]) if nObs > 0 else np.min(lon)
        preStats[k] = np.min([preStats[k],np.min(pre)]) if nObs > 0 else np.min(pre)
        timStats[k] = np.min([timStats[k],np.min(tim)]) if nObs > 0 else np.min(tim)
        uwdStats[k] = np.min([uwdStats[k],np.min(uwd)]) if nObs > 0 else np.min(uwd)
        vwdStats[k] = np.min([vwdStats[k],np.min(vwd)]) if nObs > 0 else np.min(vwd)
        nobStats[k] = np.min([nobStats[k],np.min(nob)]) if nObs > 0 else np.min(nob)
        ntyStats[k] = np.min([ntyStats[k],np.min(nty)]) if nObs > 0 else np.min(nty)
        nidStats[k] = np.min([nidStats[k],np.min(nid)]) if nObs > 0 else np.min(nid)
        qimStats[k] = np.min([qimStats[k],np.min(qim)]) if nObs > 0 else np.min(qim)
        uvrStats[k] = np.min([uvrStats[k],np.min(uvr)]) if nObs > 0 else np.min(uvr)
        vvrStats[k] = np.min([vvrStats[k],np.min(vvr)]) if nObs > 0 else np.min(vvr)
        pvrStats[k] = np.min([pvrStats[k],np.min(pvr)]) if nObs > 0 else np.min(pvr)
        tvrStats[k] = np.min([tvrStats[k],np.min(tvr)]) if nObs > 0 else np.min(tvr)
        dvrStats[k] = np.min([dvrStats[k],np.min(dvr)]) if nObs > 0 else np.min(dvr)
        qvrStats[k] = np.min([qvrStats[k],np.min(qvr)]) if nObs > 0 else np.min(qvr)
        # maxVal (*Stats[3])
        k = 3
        latStats[k] = np.max([latStats[k],np.max(lat)]) if nObs > 0 else np.max(lat)
        lonStats[k] = np.max([lonStats[k],np.max(lon)]) if nObs > 0 else np.max(lon)
        preStats[k] = np.max([preStats[k],np.max(pre)]) if nObs > 0 else np.max(pre)
        timStats[k] = np.max([timStats[k],np.max(tim)]) if nObs > 0 else np.max(tim)
        uwdStats[k] = np.max([uwdStats[k],np.max(uwd)]) if nObs > 0 else np.max(uwd)
        vwdStats[k] = np.max([vwdStats[k],np.max(vwd)]) if nObs > 0 else np.max(vwd)
        nobStats[k] = np.max([nobStats[k],np.max(nob)]) if nObs > 0 else np.max(nob)
        ntyStats[k] = np.max([ntyStats[k],np.max(nty)]) if nObs > 0 else np.max(nty)
        nidStats[k] = np.max([nidStats[k],np.max(nid)]) if nObs > 0 else np.max(nid)
        qimStats[k] = np.max([qimStats[k],np.max(qim)]) if nObs > 0 else np.max(qim)
        uvrStats[k] = np.max([uvrStats[k],np.max(uvr)]) if nObs > 0 else np.max(uvr)
        vvrStats[k] = np.max([vvrStats[k],np.max(vvr)]) if nObs > 0 else np.max(vvr)
        pvrStats[k] = np.max([pvrStats[k],np.max(pvr)]) if nObs > 0 else np.max(pvr)
        tvrStats[k] = np.max([tvrStats[k],np.max(tvr)]) if nObs > 0 else np.max(tvr)
        dvrStats[k] = np.max([dvrStats[k],np.max(dvr)]) if nObs > 0 else np.max(dvr)
        qvrStats[k] = np.max([qvrStats[k],np.max(qvr)]) if nObs > 0 else np.max(qvr)
        # update nObs
        nObs = nObs + np.size(lat)  # all vectors presumed the same length, just using lat here
    print('{:.0E} total observations processed'.format(nObs))

    # normalize *Stats[0] by float(nObs) to compute mean values
    k = 0
    latStats[k] = latStats[k] / float(nObs)
    lonStats[k] = lonStats[k] / float(nObs)
    preStats[k] = preStats[k] / float(nObs)
    timStats[k] = timStats[k] / float(nObs)
    uwdStats[k] = uwdStats[k] / float(nObs)
    vwdStats[k] = vwdStats[k] / float(nObs)
    nobStats[k] = nobStats[k] / float(nObs)
    ntyStats[k] = ntyStats[k] / float(nObs)
    nidStats[k] = nidStats[k] / float(nObs)
    qimStats[k] = qimStats[k] / float(nObs)
    uvrStats[k] = uvrStats[k] / float(nObs)
    vvrStats[k] = vvrStats[k] / float(nObs)
    pvrStats[k] = pvrStats[k] / float(nObs)
    tvrStats[k] = tvrStats[k] / float(nObs)
    dvrStats[k] = dvrStats[k] / float(nObs)
    qvrStats[k] = qvrStats[k] / float(nObs)

    # iterate through fileList
    for fileName in fileList:
        # open file-handle
        hdl = Dataset(fileName)
        # collect observations
        lat = np.asarray(hdl.variables['lat']).squeeze()
        lon = np.asarray(hdl.variables['lon']).squeeze()
        pre = np.asarray(hdl.variables['pre']).squeeze()
        tim = np.asarray(hdl.variables['tim']).squeeze()
        uwd = np.asarray(hdl.variables['uwd']).squeeze()
        vwd = np.asarray(hdl.variables['vwd']).squeeze()
        nob = np.asarray(hdl.variables['nob']).squeeze()
        nty = np.asarray(hdl.variables['nty']).squeeze()
        nid = np.asarray(hdl.variables['nid']).squeeze()
        qim = np.asarray(hdl.variables['qim']).squeeze()
        uvr = np.asarray(hdl.variables['uvr']).squeeze()
        vvr = np.asarray(hdl.variables['vvr']).squeeze()
        pvr = np.asarray(hdl.variables['pvr']).squeeze()
        tvr = np.asarray(hdl.variables['tvr']).squeeze()
        dvr = np.asarray(hdl.variables['dvr']).squeeze()
        qvr = np.asarray(hdl.variables['qvr']).squeeze()
        hdl.close()
        # accumulate variance in *Stats[1] relative to pre-computed mean in *Stats[0]
        k = 1
        j = 0
        latStats[k] = latStats[k] + np.sum((latStats[j] - lat)**2.)
        lonStats[k] = lonStats[k] + np.sum((lonStats[j] - lon)**2.)
        preStats[k] = preStats[k] + np.sum((preStats[j] - pre)**2.)
        timStats[k] = timStats[k] + np.sum((timStats[j] - tim)**2.)
        uwdStats[k] = uwdStats[k] + np.sum((uwdStats[j] - uwd)**2.)
        vwdStats[k] = vwdStats[k] + np.sum((vwdStats[j] - vwd)**2.)
        nobStats[k] = nobStats[k] + np.sum((nobStats[j] - nob)**2.)
        ntyStats[k] = ntyStats[k] + np.sum((ntyStats[j] - nty)**2.)
        nidStats[k] = nidStats[k] + np.sum((nidStats[j] - nid)**2.)
        qimStats[k] = qimStats[k] + np.sum((qimStats[j] - qim)**2.)
        uvrStats[k] = uvrStats[k] + np.sum((uvrStats[j] - uvr)**2.)
        vvrStats[k] = vvrStats[k] + np.sum((vvrStats[j] - vvr)**2.)
        pvrStats[k] = pvrStats[k] + np.sum((pvrStats[j] - pvr)**2.)
        tvrStats[k] = tvrStats[k] + np.sum((tvrStats[j] - tvr)**2.)
        dvrStats[k] = dvrStats[k] + np.sum((dvrStats[j] - dvr)**2.)
        qvrStats[k] = qvrStats[k] + np.sum((qvrStats[j] - qvr)**2.)

    # normalize *Stats[1] by float(nObs-1) to compute variance and take square-root to recover standard deviation
    k = 1
    latStats[k] = np.sqrt(latStats[k] / float(nObs-1))
    lonStats[k] = np.sqrt(lonStats[k] / float(nObs-1))
    preStats[k] = np.sqrt(preStats[k] / float(nObs-1))
    timStats[k] = np.sqrt(timStats[k] / float(nObs-1))
    uwdStats[k] = np.sqrt(uwdStats[k] / float(nObs-1))
    vwdStats[k] = np.sqrt(vwdStats[k] / float(nObs-1))
    nobStats[k] = np.sqrt(nobStats[k] / float(nObs-1))
    ntyStats[k] = np.sqrt(ntyStats[k] / float(nObs-1))
    nidStats[k] = np.sqrt(nidStats[k] / float(nObs-1))
    qimStats[k] = np.sqrt(qimStats[k] / float(nObs-1))
    uvrStats[k] = np.sqrt(uvrStats[k] / float(nObs-1))
    vvrStats[k] = np.sqrt(vvrStats[k] / float(nObs-1))
    pvrStats[k] = np.sqrt(pvrStats[k] / float(nObs-1))
    tvrStats[k] = np.sqrt(tvrStats[k] / float(nObs-1))
    dvrStats[k] = np.sqrt(dvrStats[k] / float(nObs-1))
    qvrStats[k] = np.sqrt(qvrStats[k] / float(nObs-1))
    # write *Stats to netCDF file
    ncOutFileName = userInputs.outFileName
    ncOut = Dataset(
                      ncOutFileName  , # Dataset input: Output file name
                      'w'              , # Dataset input: Make file write-able
                      format='NETCDF4' , # Dataset input: Set output format to netCDF4
                    )
    # add dimensions
    stat = ncOut.createDimension(
                                 'stat' , # nc_out.createDimension input: Dimension name
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    # add variables
    latOut = ncOut.createVariable(
                                  'lat'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    lonOut = ncOut.createVariable(
                                  'lon'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    preOut = ncOut.createVariable(
                                  'pre'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    timOut = ncOut.createVariable(
                                  'tim'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    uwdOut = ncOut.createVariable(
                                  'uwd'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    vwdOut = ncOut.createVariable(
                                  'vwd'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    nobOut = ncOut.createVariable(
                                  'nob'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    ntyOut = ncOut.createVariable(
                                  'nty'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    nidOut = ncOut.createVariable(
                                  'nid'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    qimOut = ncOut.createVariable(
                                  'qim'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    uvrOut = ncOut.createVariable(
                                  'uvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    vvrOut = ncOut.createVariable(
                                  'vvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    pvrOut = ncOut.createVariable(
                                  'pvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    tvrOut = ncOut.createVariable(
                                  'tvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    dvrOut = ncOut.createVariable(
                                  'dvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    qvrOut = ncOut.createVariable(
                                  'qvr'       ,
                                  'f8'        ,
                                  ('stat')
                                 )
    # write *Stats to *Out variables
    latOut[:] = latStats
    lonOut[:] = lonStats
    preOut[:] = preStats
    timOut[:] = timStats
    uwdOut[:] = uwdStats
    vwdOut[:] = vwdStats
    nobOut[:] = nobStats
    ntyOut[:] = ntyStats
    nidOut[:] = nidStats
    qimOut[:] = qimStats
    uvrOut[:] = uvrStats
    vvrOut[:] = vvrStats
    pvrOut[:] = pvrStats
    tvrOut[:] = tvrStats
    dvrOut[:] = dvrStats
    qvrOut[:] = qvrStats
    # close ncOut
    ncOut.close()
#
# end
#
