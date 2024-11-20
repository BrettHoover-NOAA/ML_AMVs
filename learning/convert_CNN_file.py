# A program that converts a gzipped netCDF4 CNN-file into a .npy CNN-file (compressed, fast-loading)
#   1. uncompress *.nc.gz file
#   2. read *.nc file
#   3. write to *.npy file
# The remaining (large!) *.nc file is not deleted in this program, to be performed by the user when *.npy file
# is vetted
#
import numpy as np
from netCDF4 import Dataset
import subprocess
import argparse
#
# begin
#
if __name__ == "__main__":
    # define argparser with inputs from user
    parser = argparse.ArgumentParser(description='define full-path to observation data subdirectories')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full-path to observation data subdirectories')
    parser.add_argument('anaDateTime', metavar='ANADATETIME', type=str, help='analysis date-time (YYYYMMDDHH)')
    parser.add_argument('anaHH', metavar='ANAHH', type=str, help='analysis hour (HH)')
    userInputs = parser.parse_args()
    # quality-control: if dataDir does not end in '/', add it
    dataDir = userInputs.dataDir if userInputs.dataDir[-1]=='/' else userInputs.dataDir + '/'
    # define gz_path as full-path to gzipped netCDF file
    gz_path = dataDir + userInputs.anaDateTime + '/gdas.t' + userInputs.anaHH + 'z.satwnd.tm00.bufr_d_' + userInputs.anaDateTime + '_superobs_CNN_T.nc.gz'
    # define uncompressed nc4_path as gz_path with trailing '.gz' ommitted
    nc4_path = gz_path[:-3]
    # define output npy_path as nc4_path with trailing '.nc' ommitted
    npy_path = nc4_path[:-3]
    # decompress netCDF CNN file
    subprocess.run('gunzip ' + gz_path, shell=True)
    # load CNN data
    hdl = Dataset(nc4_path)
    CNN = np.asarray(hdl.variables['CNN']).squeeze()
    # write to .npy file
    with open(npy_path + '.npy', 'wb') as f:
        np.save(f,CNN)
#
# end
#
