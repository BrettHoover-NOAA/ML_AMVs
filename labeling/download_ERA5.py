# load python dependencies (ML-labeling compliant)
import cdsapi
import datetime
import argparse
#
# define internal functions
#
# download_ERA5: for a given year, month, day, and time (as HH:MM), compute the 7 hourly periods
# of a 6-hour observation window from -3 to +3 hours relative to analysis time, and download the
# ERA5 hourly data on pressure levels from Copernicus via the CDS API. This requires setting up
# the CDS API, which includes having a registered Copernicus account and having a token identified
# in a ~/.cdsapi file.
#
# INPUTS:
#   year: analysis year in YYYY format (string)
#   month: analysis month in MM format (string)
#   day: analysis day in DD format (string)
#   time: analysis time in HH:MM format (string)
#
# OUTPUTS:
#   No explicit outputs, but downloads hourly GRIB files to local directory
#
# DEPENDENCIES
# cdsapi
def download_ERA5(year, month, day, time):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1', '2', '3',
                '5', '7', '10',
                '20', '30', '50',
                '70', '100', '125',
                '150', '175', '200',
                '225', '250', '300',
                '350', '400', '450',
                '500', '550', '600',
                '650', '700', '750',
                '775', '800', '825',
                '850', '875', '900',
                '925', '950', '975',
                '1000',
            ],
            'year': year,
            'month': month,
            'day': [day],
        'time': [time],
        'format': 'grib',
    },
    'ERA5_uv_'+year+month+day+'-H'+time+'.grib')
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directories and names of ' +
                                                 'netCDF super-ob and ERA5 GRIB files')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    # parse arguments
    userInputs = parser.parse_args()
    # define datetime object for anaDateTime
    anaDateTime = datetime.datetime.strptime(userInputs.anaDateTime, '%Y%m%d%H')
    # rewind by 3 hours to define datetime of beginning of ob-window
    begDateTime = anaDateTime - datetime.timedelta(hours=3)
    # fast forward by 3 hours to define datetime of end of ob-window
    endDateTime = anaDateTime + datetime.timedelta(hours=3)
    # begin at begDateTime
    curDateTime = begDateTime  # no deep copy required, as datetimes are immutable
    # loop through and download ERA5 data at each hour between beginning/end of ob-window
    while curDateTime <= endDateTime:
        # define inputs to download_ERA()
        curYear=datetime.datetime.strftime(curDateTime,'%Y')
        curMonth=datetime.datetime.strftime(curDateTime,'%m')
        curDay=datetime.datetime.strftime(curDateTime,'%d')
        curTime=datetime.datetime.strftime(curDateTime,'%H:%M')
        # download ERA5
        download_ERA5(curYear, curMonth, curDay, curTime)
        # increment curDateTime
        curDateTime = curDateTime + datetime.timedelta(hours=1)
#
# end
#

