"""Manage PVGIS data.

Downlaods and manages data from the PVGIS web API.
"""


import requests
import pandas as pd
import datetime as dt
import io
from .helper import req, Component
import tempfile
from functools import lru_cache
from numba import jit
import numpy as np
from typing import List


class Pvgis():
    """Manage PVIS data.

    Contains hourly data of
        - Overall pv-power per m*m,
        - Direct irradiance in W/m*m
        - Diffuse irradiance in W/m*m
        - Scattered irradiance in W/m*m
        - Windspeed @10m
        - Ambient temperature

    Attributes:
        req (str): Raw data from the PVGIS web API.
        gen_Pvgis_hourly_url (type): .
        data (pd.DataFrame): Parsed data from the response in a DataFrame.
        building ("Building"): Building the data is attached to.

    Todo:
        One access to the api for every component should be enough to
            calculate all other datasets at a given location.
        Given the expensive DataFrame.__getitem__ calls it should be possible,
            to use one Pvgis class for all components in a building.
        The initialization should be possible with a building instance to
            define all necessary parameters.

    """

    def __init__(self, loc: List[float], inclination: float = 0,
                 azimuth: float = 0, startyear: int = 2007,
                 endyear: int = 2008):
        """Initialize.

        Args:
            loc (List[float,float]): Location as list of latitude and
                longitude.
            inclination (int): inclination/angle: Inclination angle from
                horizontal plane of the (fixed) PV system. in degrees.
                Defaults to 0.
            azimuth (int): azimuth/azimuth as orientation angle of the (fixed)
                PV system, 0=south, 90=west, -90=east.
            startyear (int): Minimum year for the data. Defaults to 2007.
            endyear (int): Maximum year for the data. Defaults to 2008.

        """
        self.loc = loc
        self.inclination = inclination
        self.azimuth = azimuth
        self.req = req(self.gen_Pvgis_hourly_url(self.loc,
                                                 inclination=self.inclination,
                                                 azimuth=self.azimuth,
                                                 startyear=startyear,
                                                 endyear=endyear))
        self.data = self.req2pd(self.req)
        self.building = None
        self.cached_lookup = lru_cache(maxsize=16)(self._cached_lookup)
        self.startyear = startyear
        self.endyear = endyear
        self.last_idx = 0
        self.dates = self.data['Date']
        self.values = self.data.values[:, 1:]

    def set_data(self, new_data):
        """Set and preprocess data"""
        self.data = new_data
        self.preprocess()

    def preprocess(self):
        self.last_idx = 0
        self.dates = self.data['Date']
        self.values = self.data.values[:, 1:]

    def _cached_lookup(self, date):
        if self.dates[self.last_idx+1] == date:
            self.last_idx += 1
        else:
            self.last_idx = np.argwhere((self.dates == date).to_numpy())[0][0]
        return self.values[self.last_idx][0]
        # return self.data.loc[self.data['Date'] == date].iloc[0, 1:]

    def __getitem__(self, date: dt.datetime, debug: bool = False):
        """Return PVGIS data on given date.

        Uses linear interpolation to generate intrahour data.

        Args:
            date (dt.datetime): Date to get data on.
            debug (bool): Debugging flag.

        TODO:
            Refractor using pandas built-in method for interpolation
            Add high frequency noise to the data (clouds)

        """
        # hour = date.hour
        frac = date.minute/60
        # print(frac)
        date_ = dt.datetime(date.year, date.month, date.day, date.hour)
        curr = self.cached_lookup(date_)
        fut = self.cached_lookup(date_+dt.timedelta(hours=1))

        # curr = self.data.loc[self.data['Date']== date_].iloc[0,1:]
        # fut = self.data.loc[self.data['Date']== date_+
        #           dt.timedelta(hours=1)].iloc[0,1:]
        # sum_ = (fut*frac) + (curr*(1-frac))
        if debug:
            return fut, curr, (fut*frac), (curr*(1-frac))
        # return sum_
        return {self.data.columns[1]:
            self.mix(curr, fut, frac)}
        # return pd.Series(
        #     self.mix(curr.values.astype(np.float64),
        #              fut.values.astype(np.float64), frac),
        #              curr.index)


    def run(self):
        """Run func."""
        return self.__getitem__(self.building.date)

    @staticmethod
    def req2pd(req):
        """Read response into Dataframe."""
        data_skipfoot = b"".join(io.BytesIO(req.content).readlines()[:-12])
        return pd.read_csv(io.BytesIO(data_skipfoot), skiprows=10,
                           parse_dates=[0], date_parser=Pvgis.Pvgis_date)

    @staticmethod
    @jit(nopython=True)
    def mix(curr, fut, frac):
        """Linear interpolation."""
        return (fut*frac) + (curr*(1-frac))

    @staticmethod
    def Pvgis_date(date):
        """Return dt.datetime from PVGIS formatted date."""
        return dt.datetime.strptime(date, "%Y%m%d:%H%M")

    @staticmethod
    def gen_Pvgis_hourly_url(loc, inclination=0, azimuth=0, startyear=2007,
                             endyear=2008):
        """Return url for hourly PVGIS data."""
        lat, lon = loc[:2]
        opts = {
            "lat": [lat],
            "lon": ['+'+str(lon)],
            "raddatabase": ["PVGIS-COSMO",
                            "PVGIS-CMSAF",
                            "PVGIS-SARAH"],
            # 'browser': [1],
            "userhorizon": [""],
            "usehorizon": ["0"],
            "angle": [inclination],
            "azimuth": [azimuth],
            "startyear": [startyear],
            "endyear": [endyear],
            "mountingplace": ["building"],
            # 'optimalinclination' : [0],
            # 'optimalangles' : [0],
            "select_database_hourly": ["PVGIS-COSMO",
                                       "PVGIS-CMSAF",
                                       "PVGIS-SARAH"],
            # 'hstartyear': [2005],
            # 'hendyear': [2005],
            # 'trackingtype': [0],
            # 'hourlyangle': [0],
            # 'hourlyaspect': [0],
            # 'PVcalculation': [1],
            "pvtechchoice": ['crystSi'],
            "peakpower": [1],
            "loss": [0],
            "components": [1, 0],
        }
        url_opt = []
        for key, item in opts.items():
            if len(item) > 0:
                url_opt.append("&{}={}".format(key, item[0]))

        return ("https://re.jrc.ec.europa.eu/pvgis5/seriescalc.php?{opt}"
                ).format(opt=("".join(url_opt))[1:])

    def _old_cached_lookup(self, date):
            return self.data.loc[self.data['Date'] == date].iloc[0, 1:]

    def _old__getitem__(self, date: dt.datetime, debug: bool = False):
        """Return PVGIS data on given date.

        Uses linear interpolation to generate intrahour data.

        Args:
            date (dt.datetime): Date to get data on.
            debug (bool): Debugging flag.

        TODO:
            Refractor using pandas built-in method for interpolation
            Add high frequency noise to the data (clouds)

        """
        # hour = date.hour
        frac = date.minute/60
        # print(frac)
        date_ = dt.datetime(date.year, date.month, date.day, date.hour)
        curr = self.cached_lookup(date_)
        fut = self.cached_lookup(date_+dt.timedelta(hours=1))

        # curr = self.data.loc[self.data['Date']== date_].iloc[0,1:]
        # fut = self.data.loc[self.data['Date']== date_+
        #           dt.timedelta(hours=1)].iloc[0,1:]
        # sum_ = (fut*frac) + (curr*(1-frac))
        if debug:
            return fut, curr, (fut*frac), (curr*(1-frac))
        # return sum_

        return pd.Series(
            self.mix(curr.values.astype(np.float64),
                     fut.values.astype(np.float64), frac),
                     curr.index)