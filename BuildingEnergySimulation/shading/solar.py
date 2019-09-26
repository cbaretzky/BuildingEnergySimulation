import numpy as np
import time
import pandas as pd
import requests
import json
import geopy
import datetime as dt
import matplotlib.pyplot as plt
import io
from scipy.interpolate import interp1d
from scipy.constants import pi
from multiprocessing import Pool
from pvlib.solarposition import spa_python
from functools import lru_cache
from ..helper import req
"""
Sun position management


TODO refractor code using spa_python functionality
"""




class SunData():
    """
    Class to manage data associated with sun positions
    location loc array(2)
    horizon (array(n) or interpl1d)

    data contains an pd.Dataframe with (date | azimuth | elevation | is_up )
    """
    def __init__(self, location, horizon = None, start_year = '2006',
                end_year = '2007', timestep=6e2):
        self.location = location
        self.timestep = timestep
        self.dates = pd.date_range(str(start_year),
                                str(end_year),
                                freq="{}s".format(int(timestep)))
        if not(isinstance(horizon,interp1d )):
            self.horizon = interp1d(horizon[:,0],horizon[:,1], bounds_error=False)
        else:
            self.horizon = horizon
        suns = spa_python(self.dates, *self.location)
        suns.azimuth = suns.azimuth.map(lambda azimuth: azimuth + (azimuth<0)*360-180 )
        up = suns.elevation>self.horizon(suns.azimuth)
        suns.insert(column ="is_up",value= up, loc=0)
        self.start_year = int(start_year)
        self.end_year = (end_year)
        self.data = suns
        self.timestamp_begin = self.dates[0].timestamp()
        self.timestamp_end = self.dates[-1].timestamp()
        self.len_timeframe = self.timestamp_end - self.timestamp_begin
        self.columns= list(self.data.columns)
        self.index= self.data.index.values.astype(np.int64)*1e-9
        self.values= self.data.values
    def plot(self):
        plt.plot(self.data.azimuth[self.data.is_up==True],
                self.data.elevation[self.data.is_up==True], 'o',alpha=.01)
        plt.plot(self.horizon.x,self.horizon.y)
    def __getitem__(self, index):
        if isinstance(index,dt.datetime):
            timestamp_date = index.replace(year=self.start_year).timestamp()
            index = int((timestamp_date- self.timestamp_begin)/self.timestep)
            if not(timestamp_date == self.index[index]):
                raise IndexError(
                "Calculated index {} does not match real date {}".format(index, )
                )
            return dict(zip(self.columns, self.values[index]))
        else:
            return self.data.values[index]


class Sun():
    """
    Class to get solar information at a
    date in datetime
    loc in array(2)
    with horizon as array(azimuth, elevation)

    """
    def __init__(self, date,loc,horizon = None):
        self.date = date
        self.loc = loc
        self.altitude = None
        self.azimuth = None
        self.sun_angle()

        if isinstance(horizon, np.ndarray):
            self.horizon = horizon
            self.horizon_f = interp1d(self.horizon[:,0],self.horizon[:,1])
        elif isinstance(horizon,interp1d):
            self.horizon_f = horizon
        else:
            raise ValueError('No valid horizon data')
        self.up =self.is_up()
    @property
    def az_r(self):
        return self.azimuth*pi/180
    @property
    def al_r(self):
        return self.altitude*pi/180

    def __call__(self):
        return self.azimuth, self.altitude
    def bpy(self, az_delta=-44):
        """
        az_delta: angle between north and x axis blender
        returns sun rotation for blender
        """
        return (0, (-90+self.altitude)*pi/180,-(az_delta-self.azimuth) *pi/180)
    @property
    def az_al(self):
        return self.azimuth, self.altitude
    @property
    def vec(self):
        x = np.cos(pi-self.al_r)
        y = np.sin(self.az_r)
        z = np.sin(self.al_r)
        return np.asarray([x,y,z])

    def is_up(self):
        """
        Returns true if the sun  is above the horizon
        """
        if self.altitude<0:
            return 0
        else:
            return int(self.altitude>(self.horizon_f(self.azimuth)))
    def sun_angle(self):
        """
        Wrapper function to return sun angles as calculated by pvlib
        could be omitted if rewriting the is_up function
        """
        angles = spa_python(self.date, self.loc[0],self.loc[1])
        self.altitude = angles['elevation'][0]
        azimuth = angles['azimuth'][0]
        self.azimuth = azimuth + (azimuth<0)*360-180 #map astronomical to navigational az
        """
        old pysolar implementation
        azimuth = pysolar.solar.get_azimuth(*self.loc[:2],self.date)-180 #difference between astronomical azimuth and navigational azimuth
        altitude = max(-1,pysolar.solar.get_altitude(*self.loc[:2],self.date))

        """


def get_hz(lat,lon):
    """
    Get Horizon file at certain
    lat : 48.00
    lon : 11.000
    """
    hz_url = "http://re.jrc.ec.europa.eu/pvgis5/printhorizon.php?lat={lat}&lon={lon}&browser=1&cbhorizon1=calculated".format(**{'lat':lat, 'lon':lon})
    response = req(hz_url)
    lines  = response.content.decode('Latin-1').split('\n')
    latitude = lines[0].split()[1]
    longitude = lines[1].split()[1]
    loc = geopy.point.Point.from_string(latitude+' N '+longitude+' E ')
    data = []
    for line in lines[4:52]:
        data.append(np.asarray(list(map(float, line.split()))))
    return loc, np.vstack(data)

def gen_dates(day_res=5,minute_res=10):
    """
    Generate list of dates in 2016 with resolution in
    day_res (distance between days)
    minute_res (intraday timestep)
    (Only for sun position calculations)
    """
    start_day=dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc)
    dates = []
    for day in range(int(366/day_res)):
        curr_day=  day*day_res
        for minute in range(int(24*60/minute_res)):
            curr_minute = minute*minute_res
            dates.append(start_day+dt.timedelta(minutes=(curr_minute+curr_day*24*60)))
    return dates

def args2sun(args):
    """
    Datetime from String
    """
    date = args['date']
    loc = args['loc']
    horizon = args['horizon']
    sun = Sun(date,loc, horizon)
    return {"date" : date, "az":sun.azimuth, "el":sun.altitude, 'up':sun.is_up()}

def get_sundata(dates, loc, horizon=None):
    """
    Generate polar sun coordinates from dates
    date as datetime
    loc as array(2)[latitude, longitude]
    """
    lat = loc[0]
    lon = loc[1]
    if horizon is None:
        horizon = get_hz(loc[0],loc[1])[1]
    suns = pd.DataFrame(columns = ['date' , 'az', 'el', 'up'])
    args = [{"date" : date, "loc": loc , "horizon": horizon } for date in dates]
    with Pool(16) as p:
        suns_ = p.map(args2sun, args)
    df = pd.DataFrame(suns_)
    df.set_index("date")
    return df

def az_el2norm(az,el):
    theta = np.pi/2-el*np.pi/180
    phi = az*np.pi/180
    norm = np.asarray(
    [
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    return norm
