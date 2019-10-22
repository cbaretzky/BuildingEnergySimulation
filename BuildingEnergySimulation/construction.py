"""Construction module

Contains Building and Losses. The building is the main class to contain every
energetically relevant component of a building.

"""


import requests
import io
import pandas as pd
import datetime
from pathlib import Path
import numpy as np
from .helper import req, Component
from .shading.solar import get_hz, SunData
from .pvgis import Pvgis
from .thermal import Thermal_zone, Wall, Window
from typing import List
import geopy
import tempfile
import gzip
from hashlib import md5


class Building():
    """Configured Building.

    Contains all configuration for a selected building.

    - Location

    - Thermal losses:
    Walls, Roof, Cellar, Windows, Doors, Heater, Heat/WW Buffers,

    - Harvesters (Gains):
    SolarPV, Solarthermal cells, Windows, Groundwater/Air Heatexchanger,
    Co-generation

    - Electrical
    Battery, Inverter, Electrolysis, People

    Args:
        loc (str, list(2)): str(Address) or list(lat,lon)  of the building
            location.
        horizon (np.ndarray): Horizon elevation in deg over azimuth.
        startyear (int): minimum year of the simulation (Minimum year fetched
            from Pvgis database).
        endyear: maximum year of the simulation (Maximum year fetched
            from Pvgis database).

    Attributes:
        project_folder (str, Path): path to the project folder on disk
        timestep (int): Simulation timestep in seconds. Defaults to 600s
        startyear (int): First year of the simulationdata. Defaults to 2007
        endyear (int): Last year of the simulationdata. Defaults to 2008
        Pvgis (Pvgis): Pvgis object at building location.
        date (datetime): Datetime of the simulation.
        sundata (SunData): SunData at the building location.

    """

    def __init__(self, loc=None, horizon: np.ndarray = None,
                 startyear: int = 2007, endyear: int = 2008):
        self.project_folder = None
        self.timestep = 600
        if loc is None:
            self.location = [49.0119199, 8.4170303]
        else:
            self.location = parse_loc(loc)
        if horizon is None:
            self.horizon = get_hz(self.location[0], self.location[1])[1]
        self.startyear = startyear
        self.endyear = endyear
        self.Pvgis = Pvgis(self.location, inclination=0, azimuth=0,
                           startyear=self.startyear, endyear=self.endyear)
        self.Pvgis.set_data(self.Pvgis.data[['Date', 'Tamb']])
        self.date = self.Pvgis.data['Date'][0]
        self.sundata = SunData(self.location, self.horizon)
        self.walls: List[Wall] = []
        self.windows: List[Window] = []
        self.components = {}
        self.result = {}
        # self.result.update({'Date': self.date,
        #                    'Tamb': self.Pvgis[self.date]['Tamb']})
        self.thermal_zone = None
        Thermal_zone.reg(self)
        self.sim_results = None

    def get_component(self, searchterm: str) -> list:
        """Return all components of a specifc type.

        Args:
            searchterm (str): Name of component/type

        Returns:
            found (List): List of objects with specific name/type.

        """

        found = []
        for name, component in self.components.items():
            if searchterm in name:
                found.append(component)
        return found

    def reg(self, component, *args, **kwargs):
        """Wrapper to register from within the building instance.

        instead of::
            $ bes.Component.reg(*args, **kwargs)

        it can be::
            $ building.reg(bes.Wall, *args, **kwargs)

        """
        component.reg(self, *args, **kwargs)

    def simulate(self, timeframe_start: datetime.datetime,
                 timeframe_stop: datetime.datetime,) -> pd.DataFrame:
        """Run the simulation from timeframe_start to timeframe_stop with the
        defined timestep

        Args:
            timeframe_start (datetime.datetime): First date of timeframe.
            timeframe_stop (datetime.datetime): Last date of timeframe.
        """
        freq = '{}s'.format(self.timestep)
        times = pd.date_range(timeframe_start, timeframe_stop, freq=freq)
        self.sim_result_list = []
        for time in times:
            self.date = pd.to_datetime(time)
            self.result = {}
            self.result.update({'Date': self.date,
                                'Tamb': self.Pvgis[self.date]['Tamb']})
            for _, component in self.components.items():
                _ = component.out
            self.sim_result_list.append(self.result)
        self.sim_results = pd.DataFrame(self.sim_result_list)
        self.sim_results.index = self.sim_results.Date
        return self.sim_results


class _Losses():  # rename to operations maybe?
    """Account for all losses (Energy and Monetary) due to normal operation."""

    def __init__(self):
        raise NotImplementedError("To be implemented")
        # self.dat = pd.DataFrame()

    def reg(self, name, head):
        """
        Let Classes register new losses
        """

        pass

    def update(self, name, vals):
        """
        Shift Timestamp forward
        Do Calculations
        """
        pass


class _Cost():
    """Cost Class

    Keep track of initial investments and running costs due to
    capex and maintenance contracts.
    """

    def __init__(self):
        raise NotImplementedError("To be implemented")


def parse_loc(location_in: List[float], filecache=True):
    """Takes location parameter and returns a list of coordinates.

    This function cleans the location parameter to a list of coordinates. If
    the location_in is a list it returns the list, else it uses the geopy
    interface to generatea list of coordinates from the descriptor.
    Args:
        location_in :List[float,float], str): List of latitude and longitude

    Returns:
        loc: List[float,float]
    """

    if isinstance(location_in, str):
        if filecache:
            reqhash = md5(bytes(location_in, 'utf')).hexdigest()
            temp_dir = Path(tempfile.gettempdir())
            fname = Path(reqhash+'.geolocator_cache')
            if Path.exists(Path.joinpath(temp_dir, fname)):
                print("Using cached answer for '{}' as geolocator"
                      " request".format(location_in))
                with gzip.open(Path.joinpath(temp_dir, fname), 'rb') as f:
                    locstring = f.readlines()
                loc = [float(item.decode()) for item in locstring]
                return loc
            else:
                geolocator = geopy.geocoders.Nominatim(
                    user_agent="BuildingEnergySimulation")
                location = geolocator.geocode(location_in)
                loc = [location.latitude, location.longitude]
                with gzip.open(Path.joinpath(temp_dir, fname), 'wb') as f:
                    for coord in loc:
                        f.write(bytes(str(coord)+'\n', 'ASCII'))
                return loc
        else:

            geolocator = geopy.geocoders.Nominatim(
                user_agent="BuildingEnergySimulation")
            location = geolocator.geocode(location_in)
            loc = [location.latitude, location.longitude]
    elif isinstance(location_in, list):
        loc = location_in
        pass
    return loc
