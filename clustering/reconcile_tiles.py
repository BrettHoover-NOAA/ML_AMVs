import numpy as np
from netCDF4 import Dataset
from glob import glob
import argparse
#
# define internal functions:
#
# reconcile duplicates: Compare clusterID values between 2 lists for the same set of observations,
#                       and reduce to a single list of clusterID values with any -1 value from
#                       the first list thrown out and replaced by the equivalent value from the
#                       second list. Returns a list of indices scheduled for removal from the master
#                       array to reconcile duplicates.
#
# INPUTS:
#    clusterID1: first vector of clusterID values for observations
#    clusterID2: second vector of clusterID values for observations
#    x1: index values for finding clusterID1 obs in master vector
#    ix1: index values for finding clusterID1 obs in x1
#    x2: index values for finding clusterID2 obs in master vector
#    ix2: index values for finding clusterID2 obs in x2
#
# OUTPUTS:
#    iDump: xDump[ixDump], or the index values for finding obs in master
#           vector for removal to reconile duplicates
#
# DEPENDENCIES:
#    numpy
def reconcile_duplicates(clusterID1, clusterID2, x1, ix1, x2, ix2):
    # assume we are keeping ix1 with values clusterID1
    clusterIDKeep = clusterID1
    ixKeep = ix1
    xKeep = x1
    # assume we are removing ix2 (with values clusterID2)
    ixDump = ix2
    xDump = x2
    # find all cases of clusterIDKeep == -1
    fix = np.where(clusterIDKeep==-1)
    # swap ix1/clusterID1 with ix2/clusterID2 in [fix]:
    #   replace clusterIDKeep[fix] with clusterID2[keep]
    #   replace ixKeep[fix] with ix2[keep]
    #   replace ixDump[fix] with ix1[keep]
    clusterIDKeep[fix] = clusterID2[fix]
    ixKeep[fix] = ix2[fix]
    xKeep[fix] = x2[fix]
    ixDump[fix] = ix1[fix]
    xDump[fix] = x1[fix]
    # return xDump[ixDump]
    return xDump[ixDump]


# assign_singleton_clusters: Loop through tiles and assign ascending clusterID values to
#                            observations with a clusterID value of -1 (unassigned), which
#                            generates singleton clusters.
#
# INPUTS:
#    tileID: vector of tile ID values for each observation
#    clusterID: vector of cluster ID values for each observation
#
# OUTPUTS:
#    no explicit outputs, but modifies clusterID to assign singleton clusters a unique clusterID
#    on each tile
#
# DEPENDENCIES
#    numpy
def assign_singleton_clusters(tileID, clusterID):
    for t in np.unique(tileID):
        # find all clusterID == -1 values in tileID == t
        i = np.where((tileID==t) & (clusterID==-1))
        # find maximum clusterID value for tileID == t
        clusterMax = np.max(clusterID[np.where(tileID==t)])
        # assign ascending clusterID values to unassigned clusters in tile
        clusterID[i] = np.arange(clusterMax + 1, clusterMax + np.size(i) + 1)
    # return
    return


# shift_clusterID: Reassign clusterID values on each tile by shifting them by the maximum value
#                  of clusterID on the previous tile, to create a unique set of clusterIDs across
#                  all tiles.
#
# INPUTS:
#    tileID: vector of tile ID values for each observation
#    clusterID: vector of cluster ID values for each observation
#
# OUTPUTS:
#    no explicit output, but clusterID is modified to shift values into a unique set of
#    clusterIDs across all tiles
#
# DEPENDENCIES
#    numpy
def shift_clusterID(tileID, clusterID):
    # generate list of all unique tileID values
    tIDs = np.unique(tileID)
    # loop through all tiles except first one (leave first tile clusterIDs alone)
    for i in range(1, np.size(tIDs)):
        # locate maximum clusterID value of previous tile
        clusterMax = np.max(clusterID[np.where(tileID == tIDs[i-1])])
        # shift clusterID in tile by clusterMax + 1
        clusterID[np.where(tileID == tIDs[i])] = clusterID[np.where(tileID == tIDs[i])] + clusterMax + 1
    return
#
# begin
#
if __name__ == "__main__":
    # generate an argument parser to accept command-line inputs:
    parser = argparse.ArgumentParser(description='Define search-string for tiles, and output file-name')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full-path to data directory (ends in /)')
    parser.add_argument('searchString', metavar='SRCHSTR', type=str, help='search string up to but not including wildcard for locating tiles')
    parser.add_argument('outputNetcdfFileName', metavar='OUTFILE', type=str, help='name of output netCDF file')
    parser.add_argument('numTiles', metavar='NUMTILES', type=int, help='number of tiles that should be processed')
    userInputs = parser.parse_args()
    #userInputs = parser.parse_args(['/scratch1/NCEPDEV/stmp4/Brett.Hoover/ML_AMVs/clustering',
    #                                'gdas.t00z.satwnd.tm00.bufr_d_2023040300_Tile_',
    #                                'gdas.t00z.satwnd.tm00.bufr_d_2023040300_reconciled.nc',
    #                                30])
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    # find list of tile files to process
    tileList = glob(dataDir + userInputs.searchString + '*' + '.nc')
    tileList.sort()
    # check if tileList is the correct length, if tiles are missing report and exit
    if len(tileList) != userInputs.numTiles:
        print('ERROR: NUMBER OF TILE FILES {:d} DOES NOT MATCH EXPECTED NUMBER OF TILES {:d}, ABORTING'.format(len(tileList), userInputs.numTiles))
        raise SystemExit('num-tile mismatch error')
    print('processing {:d} tiles'.format(len(tileList)))
    # initialize empty arrays to store data across all tiles
    amvLat=np.asarray([])  # latitude (deg)
    amvLon=np.asarray([])  # longitude (deg)
    amvPre=np.asarray([])  # pressure (Pa)
    amvTim=np.asarray([])  # time (frac. hrs)
    amvUwd=np.asarray([])  # u-wind (m/s)
    amvVwd=np.asarray([])  # v-wind (m/s)
    amvTyp=np.asarray([]).astype('int')  # type (integer, categorical)
    amvIdx=np.asarray([]).astype('int')  # index (relative to input source file)
    amvPQC=np.asarray([]).astype('int')  # pre-QC (1==passed, -1==failed)
    amvTID=np.asarray([]).astype('int')  # tile-index ob was processed on
    amvCID=np.asarray([]).astype('int')  # cluster index assigned from tile (-1==unassigned)
    # process tile files, appending data to master arrays
    for tile in tileList:
        print('processing ' + tile)
        hdl = Dataset(tile)
        amvLat = np.append(amvLat, np.asarray(hdl.variables['lat']).squeeze())
        amvLon = np.append(amvLon, np.asarray(hdl.variables['lon']).squeeze())
        amvPre = np.append(amvPre, np.asarray(hdl.variables['pre']).squeeze())
        amvTim = np.append(amvTim, np.asarray(hdl.variables['tim']).squeeze())
        amvUwd = np.append(amvUwd, np.asarray(hdl.variables['uwd']).squeeze())
        amvVwd = np.append(amvVwd, np.asarray(hdl.variables['vwd']).squeeze())
        amvTyp = np.append(amvTyp, np.asarray(hdl.variables['typ']).squeeze())
        amvIdx = np.append(amvIdx, np.asarray(hdl.variables['idx']).squeeze())
        amvPQC = np.append(amvPQC, np.asarray(hdl.variables['pqc']).squeeze())
        amvTID = np.append(amvTID, np.asarray(hdl.variables['tid']).squeeze())
        amvCID = np.append(amvCID, np.asarray(hdl.variables['cid']).squeeze())
    # define unique list of amvIdx values and counts in master array (tracking duplicates)
    uniIdx, uniCnt = np.unique(amvIdx,return_counts=True)
    # loop backward through all sets of duplicates, from the maximum amount of duplication (usually 4)
    # back to the minimum (2)
    for n in range(np.max(np.unique(uniCnt)), 1, -1):
        # extract list of unique amvIdx values and their respective counts
        uniIdx, uniCnt = np.unique(amvIdx,return_counts=True)
        # extract all unique duplicate (n) entry amvIdx, along with the index to compute the unique list
        # from amvIdx
        x = np.where(np.isin(amvIdx, uniIdx[np.where(uniCnt==n)[0]]))[0]
        ux, ix = np.unique(amvIdx[x],return_index=True)
        # define x2 as all entries in amv_idx[x] not in amv_idx[x[ix]] (remaining duplicates)
        x2 = np.setdiff1d(x, x[ix])
        # extract index to compute unique list from amvIdx[x2] (another complete list of duplicates)
        ux2, ix2 = np.unique(amvIdx[x2], return_index=True)
        # define iDump, indices of observations in master arrays that need to be deleted to reconcile
        # duplicates in x, x2
        iDump = reconcile_duplicates(amvCID[x[ix]], amvCID[x2[ix2]], x, ix, x2, ix2)
        # delete iDump observations from master arrays
        amvLat = np.delete(amvLat, iDump)
        amvLon = np.delete(amvLon, iDump)
        amvPre = np.delete(amvPre, iDump)
        amvTim = np.delete(amvTim, iDump)
        amvUwd = np.delete(amvUwd, iDump)
        amvVwd = np.delete(amvVwd, iDump)
        amvTyp = np.delete(amvTyp, iDump)
        amvIdx = np.delete(amvIdx, iDump)
        amvPQC = np.delete(amvPQC, iDump)
        amvTID = np.delete(amvTID, iDump)
        amvCID = np.delete(amvCID, iDump)
        # report for clarity that 0 observations with n-duplications remain after reconciliation
        uniIdx, uniCnt = np.unique(amvIdx,return_counts=True)
        x = np.where(np.isin(amvIdx, uniIdx[np.where(uniCnt==n)[0]]))[0]
        print('{:d} obs remaining with {:d} duplicates'.format(np.size(x), n))
    # assign singleton clusterIDs
    assign_singleton_clusters(amvTID, amvCID)
    # shift clusterIDs
    shift_clusterID(amvTID, amvCID)
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
    typ = ncOut.createVariable(
                                  'typ'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    idx = ncOut.createVariable(
                                  'idx'       ,
                                  'i8'        ,
                                  ('ob')
                               )
    pqc = ncOut.createVariable(
                                  'pqc'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    tid = ncOut.createVariable(   'tid'       ,
                                  'i8'        ,
                                  ('ob')
                              )
    cid = ncOut.createVariable(
                                  'cid'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    # fill netCDF file variables
    lat[:]      = amvLat
    lon[:]      = amvLon
    pre[:]      = amvPre
    tim[:]      = amvTim
    uwd[:]      = amvUwd
    vwd[:]      = amvVwd
    typ[:]      = amvTyp
    idx[:]      = amvIdx
    pqc[:]      = amvPQC
    tid[:]      = amvTID
    cid[:]      = amvCID
    # close ncOut
    ncOut.close()
