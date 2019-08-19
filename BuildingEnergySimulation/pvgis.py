import requests
import pandas as pd
import datetime as dt
import io
from .helper import *
import tempfile
from functools import lru_cache
from numba import jit
import numpy as np

class PVGIS():
    """
    Manage PVIS data
    inclination/angle: Inclination angle from horizontal plane of the (fixed) PV system.
    azimuth/azimuth: Orientation (azimuth) angle of the (fixed) PV system, 0=south, 90=west, -90=east.
    in:
        None
    out:
        T_amb
        windspeed @10m
        pv_power
        Beam irradiance
        Diffuse irradiance
        Scattered irradiance
    """
    def cached_lookup(self, date):
        return self.data.loc[self.data['Date']== date].iloc[0,1:]
    def __init__(self, loc, inclination=0 , azimuth= 0, startyear=2007, endyear=2008):
        self.loc = loc
        self.inclination=inclination
        self.azimuth=azimuth
        self.req = req(self.gen_PVGIS_hourly_url(self.loc, inclination = self.inclination, azimuth = self.azimuth, startyear=startyear, endyear=endyear))
        self.data = self.req2pd(self.req)
        self.building = None
        self.cached_lookup = lru_cache(maxsize=16)(self.cached_lookup)
        self.startyear = startyear
        self.endyear = endyear
    def __getitem__(self, date, debug=False):
        """
        Linear PVGIS Data interpolation
        between two points

        TODO:
        refractor using pandas built-in method
        for interpolation
        Add high frequency noise to the data (clouds)
        """
        #hour = date.hour
        frac = date.minute/60
        #print(frac)
        date_ = dt.datetime(date.year, date.month, date.day, date.hour)
        curr = self.cached_lookup(date_)
        fut = self.cached_lookup( date_+dt.timedelta(hours=1))

        #curr = self.data.loc[self.data['Date']== date_].iloc[0,1:]
        #fut = self.data.loc[self.data['Date']== date_+dt.timedelta(hours=1)].iloc[0,1:]
        #sum_ = (fut*frac) + (curr*(1-frac))
        if debug:
            return fut, curr, (fut*frac) , (curr*(1-frac))
        #return sum_
        return pd.Series(
            self.mix(curr.values.astype(np.float64),fut.values.astype(np.float64),frac),
        curr.index)

    def run():
        return self.__getitem__(building.date)
    @staticmethod
    def req2pd(req):
        data_skipfoot = b"".join(io.BytesIO(req.content).readlines()[:-12])
        return pd.read_csv(io.BytesIO(data_skipfoot), skiprows=10, parse_dates=[0], date_parser=PVGIS.PVGIS_date)
    @staticmethod
    @jit(nopython=True)
    def mix(curr, fut, frac):
            return (fut*frac) + (curr*(1-frac))
    @staticmethod
    def PVGIS_date(date):
        return dt.datetime.strptime(date, "%Y%m%d:%H%M")
    @staticmethod
    def gen_PVGIS_hourly_url(loc, inclination = 0, azimuth = 0,startyear=2007, endyear=2008):
        lat ,lon = loc[:2]
        opts = {
        "startyear":[startyear],
        "endyear":[endyear],
        "raddatabase": ["PVGIS-COSMO","PVGIS-CMSAF","PVGIS-SARAH" ],
        "angle": [inclination],
        "aspect": [azimuth],
        "components":[1,0],
        }
        url_opt = []
        for key, item in opts.items():
            if len(item)>0:
                url_opt.append("&{}={}".format(key,item[0]))

        return "http://re.jrc.ec.europa.eu/pvgis5/seriescalc.php?lat={lat}&lon={lon}&peakpower=1&pvtechchoice=crystSi{opt}".format(lat=lat,lon=lon,opt="".join(url_opt))
