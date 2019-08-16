import numpy as np
import time
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt

from .helper import *
from .pvgis import *
from scipy.sparse import diags

import io


class Solar_pv_simple(Component):
    """
    Minimal solar_pv array
    
    self.kwp = kwp
    self.azimuth = azimuth
    self.elevation = inclination
    self.PVGIS = None
    self.loss = .14
    self.building = None
    self.out = ["P_el", "name"]
        
        
    out 
        p_out: Power out in W
    
    """
    @classmethod
    def reg(cls, building, kwp=10, azimuth=0, inclination=20):
        solar_pv = cls(kwp, azimuth, inclination)
        solar_pv.PVGIS = PVGIS(building.location, inclination, azimuth)
        super().__init__(solar_pv, building, connector_out = ['p_out'], )
        
        building.components.update({solar_pv.name: solar_pv})
    def __init__(self, kwp, azimuth, inclination):
        self.kwp = kwp
        self.azimuth = azimuth
        self.elevation = inclination
        self.PVGIS = None
        self.loss = .14
        self.building = None
      
    def run(self):
        date = self.building.date
        pv_power =  self.PVGIS[date]['EPV']*(1-self.loss)*self.kwp
        return {"p_out" : pv_power*self.kwp*(1-self.loss),
               "name" : self.name
               }

    
class Grid(Component):
    """
    Powergrid class, managing Bulding electrical energy
    
    self.price_in = .3 #Euro/kwh
    self.price_out = .1 #Euro/kwh
    self.p_max = 20e3 #w
    self.timestep = 600 #s
        
        
    In
        p_in int>0: Power coming from the grid into the building
        p_out int<0: Power going from the building into the grid
    Out
        cost in Euro
    """
    @classmethod
    def reg(cls, building):
        grid = cls()
        building.grid  = grid
        grid.building = building
        grid.timestep = building.timestep
        super().__init__(grid, building,connector_in = ['p_out','p_in'], connector_out = ['cost'], )
        building.components.update({grid.name :grid})

    def __init__(self):
        self.price_in = .3 #Euro/kwh
        self.price_out = .1 #Euro/kwh
        self.p_max = 20e3 #w
        self.timestep = 600
    def run(self, p_out = 0., p_in =0., merge=False ):
        """
        p_out
        """
        self.p_out=  p_out
        self.p_in = p_in
        if self.p_out > self.p_max:
            print("Maximum grid power exceeded")
        if merge == False:
            self.cost = p_out*self.price_out*self.timestep/(3.6e6)+p_in*self.timestep/(3.6e6)*self.price_in
        else:
            price = self.price_in-int((p_in-p_out)>0)*(self.price_in-self.price_out)
            self.cost = (self.p_out-self.p_in)*pricea
                
        return {'cost' : self.cost,
                'name' : self.name}

            
class Battery(Component):
    """
    Simple Battery model:

    
    self.capacity = 13e3*3600  #usable capacity in J
    self.charge = 0
    self.timestep = 600
    self.efficiency
    self.p_max = self.capacity /3600/2 # Max power, full discharge in 2 hours
    self.cycle=0  #in J
    self.from_grid=0 #in W
    self.to_grid=0 #in W
    In
        p_out int>0: Power coming from the building into the battery
        p_in int<0: Power going from the battery into the building
    Out
        p_out int>0: Overflow from the battery
        p_in int<0: Demand to cover power needs
        (Cash due to ageing loss)
    TODO:
    Implement ageing loss
    ref. Ageing of Lithium-Ion Batteries in Electric Vehicles (Dipl.-Ing. Univ. Peter Keil)
    
    """
    @classmethod
    def reg(cls, building, capacity=468e5, ):
        battery = cls(capacity)
        battery.building = building
        battery.timestep =building.timestep
        super().__init__(battery, building,connector_in = ['p_out','p_in'], connector_out = ['p_out','p_in','p_sum','cycle'] , )
        building.components.update({battery.name :battery})
    
    def __init__(self, capacity=468e5, name=None):
        self.capacity = capacity #capacity in joule
        self.charge = .5*self.capacity #Current charge in joule
        self.timestep = 600
        self.efficiency = .95 #Store/retrieve efficiency
        self.p_max = self.capacity /3600/2 # Max power, full discharge in 2 hours
        self.cycle=0  #in J
        self.from_grid=0 #in J Energy to Grid (>0)
        self.to_grid=0 #in J Energy from grid (<0)
        
    def run(self, p_in=0., p_out=0.):
        """
        Given a small enough timestep, the assumption is that input energy and output energy can be
        summed up against each other. If the timestep gets longer this assumption will get worse, as it 
        potentially assumes a perfect energy storage during the timestep.
        """
        old_charge = self.charge
        p_sum = (p_in+p_out)
        self.from_grid = 0.
        self.to_grid = 0.
        if p_sum>0:
            #charge regime
            if abs(p_sum) > self.p_max:
                p = self.p_max
                self.from_grid = (p_sum-self.p_max) #overflow to grid due to max power
            else:
                p=p_sum
            temp_charge = self.charge+p*self.timestep*self.efficiency
            if temp_charge>self.capacity:
                self.to_grid+= (temp_charge-self.capacity)/self.timestep #overflow to grid due to max capacity
                self.charge = self.capacity
            else:
                self.charge = temp_charge
        elif p_sum<0:
            #discharge regime
            if abs(p_sum) > self.p_max:
                p = -self.p_max
                self.to_grid = (p_sum+self.p_max) #overflow to grid due to max power
            else:
                p=p_sum
            temp_charge = self.charge+p*self.timestep*1/self.efficiency
            if temp_charge < 0:
                self.from_grid+= temp_charge*self.efficiency/self.timestep #
                self.charge = 0.
            else:
                self.charge = temp_charge
        else:
            #No power going through the system
            pass
        self.cycle+=abs(self.charge-old_charge)/self.capacity/2
        
        
        return {'p_in' : self.from_grid,
                'p_out' : self.to_grid,
                'p_sum' : p_sum,
                'charge' : self.charge,
                'cycle' : self.cycle,
                'name' : self.name}
        
        
class Inverter():
    """
    Configure inverter with efficiency curve eta/P
    
    using quadratic loss model
    
    #config as dict with Unit in Watts
    self.p_mpp         #Losses from the running tracker (const)
    self.p_standby     #Standby losses (const)
    self.hysteresis    #Linear losses (linear)
    self.r_ohm_div_u   #Ohmic losses (squared)
    self.p_min         #Minimal operating power (in)
    self.p_max         #Maximal power (in)
    
    Todo:
    Peakpower/Maxpower, temporary exceeding limitations
    Better Loss Model
    Multiple Strings
    Base on Voltage/Current, not Power
    
    self.cost
    """
    def __init__(self, config={}):
        self.p_mpp = 100
        self.p_standby = 10
        self.r_ohm_div_u = 1e-5
        self.hysteresis = 1e-2
        self.p_min = 1e2
        self.p_max = 1e4
        self.cost = 2e2
        self.__dict__.update(config)
    
    def p_out(self, p_in):
        p = np.minimum(np.asarray(p_in))
        return p*(p>self.p_min)-(self.p_standby+self.p_mpp*(p>self.p_min))\
                   -self.hysteresis*p*(p>self.p_min)\
                   -self.r_ohm_div_u*p**2*(p>self.p_min)
    def plot_efficiency(self):
        p= np.linspace(0,self.p_max,100)
        po = self.p_out(p)
        
        best_efficiency=  np.max(po/p)
        optimal_power=  p[np.argmax(po/p)]
        max_power_efficiency = (po/p)[-1]
        print("{} % efficiency at {:.2f} W".format(best_efficiency, optimal_power ))
        print("{} % efficiency at {:.2f} W".format(max_power_efficiency, self.p_max))
        plt.plot(po/p,'+-')

        
class Dc_dc():
    """
    Simple Dc-Dc Converter to help with
    dc-coupled systems
    
    efficiency: linear efficiency factor
    p_max: maximum output power = set 0 if infinite
    """
    def __init__(efficiency = .97, p_max = 0):
        self.p_max = 0
        self.efficiency = .97
    def p_out(self, p_in):
        if self.p_max:
            return np.minimum(p_in*self.efficiency, self.p_max)
        else: 
            return p_in*self.efficiency
            
            
class Inverter_sandia():
    """
    Configure inverter with efficiency curve eta/P
    Max Power Conversion Pmax
    Following on https://energy.sandia.gov/wp-content/gallery/uploads/Performance-Model-for-Grid-Connected-Photovoltaic-Inverters.pdf
    page 14
    Pac = {(Paco / (A - B)) – C ⋅ (A - B)}⋅ (Pdc- B) + C ⋅ (Pdc - B)**2
    A = Pdco⋅{1 + C1⋅(Vdc - Vdco)}    
    B = Pso⋅{1 + C2⋅(Vdc - Vdco)}   
    C = Co⋅{1 + C3⋅(Vdc - Vdco)}  
    
    Pac    =    ac-power output from inverter based on input power and voltage, (W) 
    Pdc    =    dc-power input to inverter, typically assumed to be equal to the PV array maximum power, (W) 
    Vd     =    dc-voltage input, typically assumed to be equal to the PV array maximum power voltage, (V) 
    Paco   =    maximum ac-power “rating” for inverter at reference or nominal operating condition, assumed to be an upper limit value, (W) 
    Pdco   =    dc-power level at which the ac-power rating is achieved at the reference operating condition, (W) 
    Vdco   =    dc-voltage level at which the ac-power rating is achieved at the reference operating condition, (V) 
    Pso    =    dc-power required to start the inversion process, or self-consumption by inverter, strongly influences inverter efficiency at low power levels, (W) 
    Pnt    =    ac-power consumed by inverter at night (night tare) to maintain circuitry required to sense PV array voltage, (W) 
    C0     =    parameter defining the curvature (parabolic) of the relationship between ac-power and dc-power at the reference operating condition, 
                default value of zero gives a linear relationship, (1/W) 
    C1     =    empirical coefficient allowing Pdco to vary linearly with dc-voltage input, default value is zero, (1/V) 
    C2     =    empirical coefficient allowing Pso to vary linearly with dc-voltage input, default value is zero, (1/V) 
    C3     =    empirical coefficient allowing Co to vary linearly with dc-voltage input, default value is zero, (1/V)
    
    """
    def __init__(self, config):
        self.p_ac 
        self.p_dc
        self.v_d
        self.p_aco
        self.p_dco
        self.v_dco
        self.p_so
        self.p_nt
        self.c0
        self.c1
        self.c2
        self.c3
        self.v_dc
        self.__dict__.update(config)
        
        
    def efficiency(self, P_out):
        return ((self.p_aco / (self.A - self.B)) - \
                self.C * (self.A - self.B))* (self.p_dc- self.B) + \
                self.C * (self.p_dc - self.B)**2
    def A(self):
        return self.p_dco*(1+self.c1*(self.v_dc))
    def B(self):
        return self.p_so*{1 + self.c2*(self.v_dc - self.v_dco)}     
    def C(self):
        return self.c0*{1 + self.c3*(self.v_dc - self.v_dco)}   