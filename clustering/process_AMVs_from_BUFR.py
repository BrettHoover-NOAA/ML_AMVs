import numpy as np
import process_satwnds_dependencies
from netCDF4 import Dataset
import datetime
import argparse
#
# define internal functions
#
# spddir_to_uwdvwd: Given a vector of wind speed and direction, return vectors of the u- and v-
#                   components.
# INPUTS:
#    wspd: vector of wind speeds (float, probably m/s)
#    wdir: vector of wind directions (float, deg)
#
# OUTPUTS:
#    uwd: vector of wind u-components
#    vwd: vector of wind v-components
#
# DEPENDENCIES
#    numpy
def spddir_to_uwdvwd(wspd, wdir):
    degToRad = np.pi/180.
    uwd = np.multiply(-wspd, np.sin(wdir*(degToRad)))
    vwd = np.multiply(-wspd, np.cos(wdir*(degToRad)))
    return uwd, vwd


# generate_date: Given a year, month, and day, produce a datetime object
#                with the corresponding date, setting the hour and minute to zero.
#
# INPUTS:
#    year: year (int)
#    month: month (int)
#    day: day (int)
#
# OUTPUTS
#    datetime object of (year, month, day, 0, 0)
#
# DEPENDENCIES:
#
#    datetime
def generate_date(year, month, day):
    return datetime.datetime(year, month, day, 0, 0)


# add_time: Given a datetime and the hour and minute, adjust the datetimeby the chosen time.
#
# INPUTS:
#    dt: datetime object
#    hour: hour (int)
#    minute: minute (int)
#
# OUTPUTS:
#    dt, adjusted forward by chosen hour and minute values
#
# DEPENDENCIES:
#    datetime 
def add_time(dt, hour, minute):
    return dt + datetime.timedelta(hours=int(hour)) + datetime.timedelta(minutes=int(minute))


# define_delt: Given a datetime and epoch datetime, compute time-difference (datetime minus epoch)
#
# INPUTS:
#    dt_epoch: datetime of epoch
#    dt: datetime
#
# OUTPUTS:
#    fractional hours between datetime and epoch (float)
#
# DEPENDENCIES:
#    datetime
def define_delt(dt_epoch, dt):
    td = dt - dt_epoch
    return (td.total_seconds())/3600.


#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directory, name of ' +
                                                 'BUFR AMV file, and name of output netCDF file')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full path to data directory, ending in /')
    parser.add_argument('bufrFileName', metavar='BUFRFILE', type=str, help='name of input BUFR AMV file')
    parser.add_argument('netcdfFileName', metavar='INFILE', type=str, help='name of output netCDF AMV file')
    # parse arguments
    userInputs = parser.parse_args()
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    tankNameList = ['NC005030',
                    'NC005031',
                    'NC005032',
                    'NC005034',
                    'NC005039',
                    'NC005044',
                    'NC005045',
                    'NC005046',
                    'NC005067',
                    'NC005068',
                    'NC005069',
                    'NC005070',
                    'NC005071',
                    'NC005072', 
                    'NC005080',
                    'NC005081',
                    'NC005091'
                   ]
    # initialize empty arrays
    obSID = np.asarray([])
    obLat = np.asarray([])
    obLon = np.asarray([])
    obPre = np.asarray([])
    obSpd = np.asarray([])
    obDir = np.asarray([])
    obUwd = np.asarray([])
    obVwd = np.asarray([])
    obYr  = np.asarray([])
    obMon = np.asarray([])
    obDay = np.asarray([])
    obHr  = np.asarray([])
    obTim = np.asarray([])
    obMin = np.asarray([])
    obTyp = np.asarray([])
    obQIn = np.asarray([])
    obPQC = np.asarray([])
    # define analysis datetime
    anDatetime = datetime.datetime.strptime(userInputs.anaDateTime, '%Y%m%d%H')
    for tankName in tankNameList:
        print('processing ' + tankName)
        outDict={
                 tankName + '/SAID'        : 'satelliteID',
                 tankName + '/CLAT'        : 'latitude',
                 tankName + '/CLON'        : 'longitude',
                 tankName + '/PRLC'        : 'pressure',
                 tankName + '/WSPD'        : 'windSpeed',
                 tankName + '/WDIR'        : 'windDirection',
                 tankName + '/YEAR'        : 'year',
                 tankName + '/MNTH'        : 'month',
                 tankName + '/DAYS'        : 'day',
                 tankName + '/HOUR'        : 'hour',
                 tankName + '/MINU'        : 'minute'
                }
        bufrFileName = dataDir + userInputs.bufrFileName
        # attempt to extract data from tank, exceptions raise warning and do not append tank data
        try:
            amvDict = process_satwnds_dependencies.process_satwnd_tank(tankName, bufrFileName, outDict)
            # append data to master arrays
            obSID = np.append(obSID, amvDict['satelliteID'])
            obLat = np.append(obLat, amvDict['latitude'])
            obLon = np.append(obLon, amvDict['longitude'])
            obPre = np.append(obPre, amvDict['pressure'])
            obSpd = np.append(obSpd, amvDict['windSpeed'])
            obDir = np.append(obDir, amvDict['windDirection'])
            obYr  = np.append(obYr,  amvDict['year'])
            obMon = np.append(obMon, amvDict['month'])
            obDay = np.append(obDay, amvDict['day'])
            obHr  = np.append(obHr,  amvDict['hour'])
            obMin = np.append(obMin, amvDict['minute'])
            obTyp = np.append(obTyp, amvDict['observationType'])
            obPQC = np.append(obPQC, amvDict['preQC'])
            obQIn = np.append(obQIn, amvDict['qualityIndicator'])
        except:
            print('warning: ' + tankName + ' was not processed due to errors')
    # derive computed variable-types for all retrieved observations
    # (1) obTim:
    # (1a) compute dates (year, month, day)
    obDates = list(map(generate_date, obYr.astype('int'), obMon.astype('int'), obDay.astype('int')))
    # (1b) compute datetimes (year, month, day, hour, minute) from dates and obHour, obMin
    obDatetimes = np.asarray(list(map(add_time, obDates, obHr.astype('int'), obMin.astype('int'))))
    # (1c) compute fractional hours relative to analysis-time, as obTim
    obTim = np.asarray(list(map(define_delt, np.repeat(anDatetime, np.size(obDatetimes)), obDatetimes))).squeeze()
    # (2) obUwd and obVwd:
    # (2a) compute obUwd and obVwd from obSpd, obDir
    obUwd, obVwd = spddir_to_uwdvwd(obSpd, obDir.astype('float'))
    # (3) change any non-(1,-1) value obPQC to (-9999), to flag any missing-data values that may appear
    obPQC[np.where((obPQC != 1.) & (obPQC != -1.))] = -9999.
    # report ob-types and pre-QC
    for t in np.unique(obTyp):
        i = np.where(obTyp==t)
        n = np.size(i)
        p = np.size(np.where(obPQC[i]==1.))
        f = np.size(np.where(obPQC[i]==-1.))
        print('{:d} observations of Type={:d} ({:.1f}% pass pre-QC, {:.1f}% fail)'.format( n,
                                                                                           int(t),
                                                                                           100. * float(p)/float(n),
                                                                                           100. * float(f)/float(n)
                                                                                          ))
    # save data to netCDF file
    nc_out_filename = dataDir + userInputs.netcdfFileName
    nc_out = Dataset( 
                      nc_out_filename  , # Dataset input: Output file name
                      'w'              , # Dataset input: Make file write-able
                      format='NETCDF4' , # Dataset input: Set output format to netCDF4
                    )
    # Dimensions
    ob = nc_out.createDimension( 
                                 'ob' , # nc_out.createDimension input: Dimension name 
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    # Variables
    lat = nc_out.createVariable(
                                  'lat'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    lon= nc_out.createVariable(
                                  'lon'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    pre = nc_out.createVariable(
                                  'pre'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    wspd = nc_out.createVariable(
                                  'wspd'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    uwd = nc_out.createVariable(
                                  'uwd'       ,
                                  'f8'         ,
                                  ('ob')
                                 )
    vwd = nc_out.createVariable(
                                  'vwd'       ,
                                  'f8'         ,
                                  ('ob')
                                 )
    wdir = nc_out.createVariable(
                                  'wdir'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    year = nc_out.createVariable(
                                  'year'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    mon = nc_out.createVariable(
                                  'mon'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    day = nc_out.createVariable(
                                  'day'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    hour = nc_out.createVariable(
                                  'hour'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    minute = nc_out.createVariable(
                                   'minute'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    tim = nc_out.createVariable(
                                  'tim'      ,
                                  'f8'        ,
                                  ('ob')
                                )
    typ = nc_out.createVariable(
                                  'typ'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    pqc = nc_out.createVariable(
                                  'pqc'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    qin = nc_out.createVariable(
                                  'qin'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    sid = nc_out.createVariable(
                                  'sid'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    # Fill netCDF file variables
    lat[:]      = obLat
    lon[:]      = obLon
    pre[:]      = obPre
    wspd[:]     = obSpd
    wdir[:]     = obDir.astype('float')
    uwd[:]      = obUwd
    vwd[:]      = obVwd
    year[:]     = obYr.astype('int')
    mon[:]      = obMon.astype('int')
    day[:]      = obDay.astype('int')
    hour[:]     = obHr.astype('int')
    minute[:]   = obMin.astype('int')
    tim[:]      = obTim
    typ[:]      = obTyp.astype('int')
    pqc[:]      = obPQC.astype('int')
    qin[:]      = obQIn.astype('int')
    sid[:]      = obSID.astype('int')
    # Close netCDF file
    nc_out.close()
#
# end
#
