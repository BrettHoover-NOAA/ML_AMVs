import cdsapi
import datetime

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



year=2023
month=1
day=1
hour=0  # hour of analysis, 6-hour observation window is centered on this time

anaDateTime = datetime.datetime(year,month,day,hour)

begDateTime = anaDateTime - datetime.timedelta(hours=3)
endDateTime = anaDateTime + datetime.timedelta(hours=3)

curDateTime = begDateTime  # no deep copy required, as datetimes are immutable
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

