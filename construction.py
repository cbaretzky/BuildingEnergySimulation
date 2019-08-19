import requests
import io
import pandas as pd
import numpy as np
from .helper import *
from .solar import *
from .pvgis import *
from .electrical import *
from .thermal import *

def ret_default_config(loc):
    """
    Return Simple building model to output Data
    Takes
    loc as str or list to use address(string) or coordinates(list/ndarray(2))
    """
    pass




class Building():
    """
    Configured Building:
    Location?
    Thermal losses (Outer Hull):
        Walls, Roof, Cellar, Windows, Doors, Heater, Heat/WW Buffers, Co-generation
    Harvesters (Gains):
        SolarPV, Solarthermal cells, Windows, Groundwater/Air Heatexchanger
    Electrical:
        Battery, Inverter, Electrolysis
    """
    def __init__(self, loc=None, horizon = None, startyear = 2007, endyear = 2008, config =[]):
        """
        Setup External parameters
        """
        self.__dict__.update(config)
        self.project_folder = None
        self.timestep = 600
        if loc==None:
            self.location = [49.0119199,8.4170303]
        else:
            self.location = parse_loc(loc)
        if horizon == None:
            self.hz_data = get_hz(self.location[0],self.location[1])[1]
        self.PVGIS = PVGIS(self.location, inclination = 0, azimuth= 0)
        self.date = self.PVGIS.data['Date'][0]
        #self.sundata = SunData(self.location, self.hz_data)
        self.walls = []
        self.windows = []
        self.components = {}
        self.result = {}
        self.result.update({'Date' : self.date,
                               'Tamb' : self.PVGIS[self.date]['Tamb']})
        self.curr_row = None
        self.thermal_zone = None
        Thermal_zone.reg(self)
        self.sim_results = None
    def get_component(self, searchterm):
        """
        Return all components of a specifc type from the registered components
        """
        found = []
        for name, component in self.components.items():
            if searchterm in name:
                found.append(component)
        return found
    def reg(self,component, *args, **kwargs):
        """
        Wrapper to register from within the building instance
        instead of bes.Component.reg(*args, **kwargs) it can be
        building.reg(bes.Wall, *args, **kwargs)

        """
        component.reg(self, *args, **kwargs)

    def simulate(self, timeframe_start, timeframe_stop,):
        """
        Run the simulation from timeframe_start to timeframe_stop with the defined timestep
        """
        freq = '{}s'.format(self.timestep)
        times = pd.date_range(timeframe_start, timeframe_stop, freq=freq)
        self.sim_result_list = []
        for time in times:
            self.date = pd.to_datetime(time)
            self.result = {}
            self.result.update({'Date' : self.date,
                               'Tamb' : self.PVGIS[self.date]['Tamb']})
            for name, component in self.components.items():
                _ = component.out
            self.sim_result_list.append(self.result)
        self.sim_results = pd.DataFrame(self.sim_result_list)
        self.sim_results.index = self.sim_results.Date
        return sim_results



class Losses(): #rename to operations maybe?
    """
    Account for all losses (Energy and Monetary) due to normal operation
    """
    def __init__(self):
        raise NotImplementedError("To be implemented")
        self.dat = pd.DataFrame()
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


class Cost():
    """
    Keep track of initial investments and running costs due to capex and maintenance contracts.
    """
    def __init__(self):
        raise NotImplementedError("To be implemented")

def parse_loc(location_in):
    """
    Parse location
    """
    geolocator = geopy.geocoders.Nominatim(user_agent = "BuildingEnergySimulation")
    if isinstance(location_in,str):
        location = geolocator.geocode(location_in)
        loc = [location.latitude, location.longitude]
    elif isinstance(location_in, list):
        loc = location_in
        pass
    return loc
