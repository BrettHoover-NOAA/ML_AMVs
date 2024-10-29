# load python dependencies (ML-labeling compliant)
import requests
import datetime
import argparse
#
# define internal functions
#
# download_ERA5: for a given year, month, day, and time (as HH:MM), compute the hourly periods
# of a 6-hour observation window at -3, 0, and +3 hours relative to analysis time, and download the
# ERA5 hourly data on pressure levels from NCAR RDA-CISL, which provides ERA5 wind components at hourly
# intervals in daily files (00-23 hrs).
#
# INPUTS:
#   dayTimeStamp: day-file time-stamp in YYYYMMDD00_YYYYMMDD23 format (string)
#
# OUTPUTS:
#   No explicit outputs, but downloads required GRIB files to local directory
#
# DEPENDENCIES
# requests
def download_ERA5_CISL(dayTimeStamp):
    dayTimeYYYYMM = dayTimeStamp[0:6]
    files = [
        "e5.oper.an.pl/" + dayTimeYYYYMM + "/e5.oper.an.pl.128_131_u.ll025uv." + dayTimeStamp + ".nc",
        "e5.oper.an.pl/" + dayTimeYYYYMM + "/e5.oper.an.pl.128_132_v.ll025uv." + dayTimeStamp + ".nc"
    ]
    # download the data file(s)
    for file in files:
        idx = file.rfind("/")
        if (idx > 0):
            ofile = file[idx+1:]
        else:
            ofile = file
        response = requests.get("https://data.rda.ucar.edu/d633000/" + file)
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
    # rewind by 3 hours to define datetime of beginning of ob-window
    begDateTime = anaDateTime - datetime.timedelta(hours=3)
    # fast forward by 3 hours to define datetime of end of ob-window
    endDateTime = anaDateTime + datetime.timedelta(hours=3)
    # collect all time-stamps for input files into a list
    # NOTE: all time-stamps are of the form YYYYYMMDD00_YYYYMMDD23, may be the same between any or all required datetimes
    timeStampList = []
    timeStampList.append(begDateTime.strftime('%Y') + begDateTime.strftime('%m') +
                         begDateTime.strftime('%d') + '00_' + begDateTime.strftime('%Y') +
                         begDateTime.strftime('%m') + begDateTime.strftime('%d') + '23')
    timeStampList.append(anaDateTime.strftime('%Y') + anaDateTime.strftime('%m') +
                         anaDateTime.strftime('%d') + '00_' + anaDateTime.strftime('%Y') +
                         anaDateTime.strftime('%m') + anaDateTime.strftime('%d') + '23')
    timeStampList.append(endDateTime.strftime('%Y') + endDateTime.strftime('%m') +
                         endDateTime.strftime('%d') + '00_' + endDateTime.strftime('%Y') +
                         endDateTime.strftime('%m') + endDateTime.strftime('%d') + '23')
    # reduce timeStampList to only unique values using set()
    timeStampList = list(set(timeStampList))
    # for each timeStamp in timeStampList, download files from NCAR RDA-CISL
    for timeStamp in timeStampList:
        download_ERA5_CISL(timeStamp)
#
# end
#
