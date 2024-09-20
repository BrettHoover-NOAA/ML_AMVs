# load python dependencies (ML-labeling compliant)
import requests
import datetime
import argparse
#
# define internal functions
#
# download_ERA5: for a given timestamp representing a 6-hour period, download the
# ERA5 hourly data on model levels from NCAR RDA-CISL, which provides ERA5 wind components at hourly
# intervals in 6-hourly files (00-05 hrs, 6-11 hrs, 12-17 hrs, 18-23 hrs).
#
# INPUTS:
#   dayTimeStamp: 6-hourly-file time-stamp in YYYYMMDDHH_YYYYMMDDHH format (string)
#
# OUTPUTS:
#   No explicit outputs, but downloads required GRIB files to local directory
#
# DEPENDENCIES
# requests
def download_ERA5_CISL(dayTimeStamp):
    dayTimeYYYYMM = dayTimeStamp[0:6]
    files = [
        "e5.oper.an.ml/" + dayTimeYYYYMM + "/e5.oper.an.ml.0_5_0_0_0_t.regn320sc." + dayTimeStamp + ".nc",  # temperature on model-levels
        "e5.oper.an.ml/" + dayTimeYYYYMM + "/e5.oper.an.ml.128_134_sp.regn320sc." + dayTimeStamp + ".nc"    # surface-pressure
    ]
    # download the data file(s)
    for file in files:
        idx = file.rfind("/")
        if (idx > 0):
            ofile = file[idx+1:]
        else:
            ofile = file
        response = requests.get("https://data.rda.ucar.edu/ds633.6/" + file)
        with open(ofile, "wb") as f:
            f.write(response.content)
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define analysis datetime')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    # parse arguments
    userInputs = parser.parse_args()
    # define datetime object for anaDateTime
    anaDateTime = datetime.datetime.strptime(userInputs.anaDateTime, '%Y%m%d%H')
    # rewind by 1 to 6 hours to define datetime of before-period
    beforeDateTime1 = anaDateTime - datetime.timedelta(hours=6)
    beforeDateTime2 = anaDateTime - datetime.timedelta(hours=1)
    # fast forward by 0-5 hours to define datetime of during-period
    duringDateTime1 = anaDateTime
    duringDateTime2 = anaDateTime + datetime.timedelta(hours=5)
    # fast forward by 6-11 hours to define datetime of during-period
    afterDateTime1 = anaDateTime + datetime.timedelta(hours=6)
    afterDateTime2 = anaDateTime + datetime.timedelta(hours=11)
    # collect all time-stamps for input files into a list
    # NOTE: all time-stamps are of the form YYYYYMMDDH1_YYYYMMDDH2, may be the same between any or all required datetimes
    timeStampList = []
    timeStampList.append(beforeDateTime1.strftime('%Y%m%d%H') + '_' + beforeDateTime2.strftime('%Y%m%d%H'))
    timeStampList.append(duringDateTime1.strftime('%Y%m%d%H') + '_' + duringDateTime2.strftime('%Y%m%d%H'))
    timeStampList.append(afterDateTime1.strftime('%Y%m%d%H') + '_' + afterDateTime2.strftime('%Y%m%d%H'))
    # for each timeStamp in timeStampList, download files from NCAR RDA-CISL
    for timeStamp in timeStampList:
        download_ERA5_CISL(timeStamp)
#
# end
#
