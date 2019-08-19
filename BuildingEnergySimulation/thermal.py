#TODO:
#replace materials with best practice global variable name.

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
from scipy.constants import zero_Celsius

DEFAULT_MATERIALS = [
 {'rho': 830.0, 'c_v': 840.0, 'lambda': 0.0285, 'name': 'Insulation Glass'},
 {'rho': 1800.0, 'c_v': 1000.0, 'lambda': 0.79, 'name': 'Brick'},
 {'rho': 2600.0,'c_v': 1000.0,'lambda': 2.8,'name': 'Granite'},
 {'rho': 35.0, 'c_v': 2100.0, 'lambda': 0.04, 'name': 'Woodfiber'},
 {'rho': 470.0, 'c_v': 1600.0, 'lambda': 0.13, 'name': 'Pine'}]



DEFAULT_LAYERS = {
    'Brick_Granite' : [{'c_v': 1000.0,
  'lambda': 0.79,
  'name': 'Brick',
  'rho': 1800.0,
  'thickness': 0.3},
 {'c_v': 1000.0,
  'lambda': 2.80,
  'name': 'Granite',
  'rho': 2600.0,
  'thickness': 0.4}], 
    'Brick' : [{'c_v': 1000.0,
  'lambda': 0.79,
  'name': 'Brick',
  'rho': 1800.0,
  'thickness': 0.3}],
    'Window' :[{'c_v': 840.0,
  'lambda': 0.0285,
  'name': 'Insulation Glass',
  'rho': 830.0,
  'thickness': 0.036}],
    'Roof' : [{'c_v': 2100.0,
  'lambda': 0.04,
  'name': 'Woodfiber',
  'rho': 35.0,
  'thickness': 0.3}]
    
}
class Material_db():
    def __init__(self):
        self.materials = pd.DataFrame.from_dict(MATERIALS_)

    def load_material_db(self, path='materials_de.csv'):
        """
        Load material database from path
        """
        materials = pd.read_csv(path)
        try:
            self.materials = materials.drop(columns ='Unnamed: 0')
        except:
            pass

    def find_material(self, name):
        if self.materials is None:
            raise ValueError("No materials loaded")
        return (self.materials[self.materials['name'].str.contains(name, case=False, na=False)])

    @staticmethod
    def gen_layers(materials, thicknesses):

        """
        Generates a layer dictionary from  a
        list of materials and corresponding thicknesses.

        Ordered from the inside out. (The first material is 
        the innermost layer)
        """
        layers = []
        for material, thickness in zip(materials, thicknesses):
            mat_dict = material.to_dict(orient='records')[0]
            mat_dict.update({'thickness' : thickness})
            layers.append(mat_dict
            )
        return layers

class Thermal_zone(Component):
    """
    Manage thermal energy flow through the building
    """
    @classmethod
    def reg(cls, building, T_in = 294 ):
        thermal_zone = cls(building, T_in = T_in) 
        super().__init__(thermal_zone, building, connector_in = ['phi'], connector_out = ['phi'])
        thermal_zone.building  = building
        building.components.update({
            thermal_zone.name: thermal_zone
        })
        building.thermal_zone = thermal_zone
    def __init__(self, building, T_in = 294, description = None):
        self.building = building
        self.walls = building.walls
        self.windows= building.windows
        self._T_in = T_in #Temperature inside the building
        self.description = description
    
    @property
    def T_in(self):
        return self._T_in
    
    @property
    def T_out(self):
        return self.building.PVGIS[self.building.date]['Tamb']+zero_Celsius
    def run(self, phi):
        """
        Returns the sum of thermal flux in and out of the building
        phi_sum > 0 means thermal energy is transfered in 
        phi_sum < 0 means thermal energy is transfered out
        """
        if phi >0:
            phi = 0 #No heating buffer through the rooms as a good radiator model is not integrated yet
        return {"phi" : phi,}
    def run_old(self, phi):
        """
        Returns the sum of thermal flux in and out of the building
        phi_sum > 0 means thermal energy is transfered in 
        phi_sum < 0 means thermal energy is transfered out
        """
        phi_sum = 0
        phis = []
        for wall in self.walls:
            phis.append(wall.run(self.T_in, self.T_out))
        for window in self.windows:
            phis.append(window.run())
        for phi in phis:
            phi_sum += phi['phi']
        if phi_sum >0:
            phi_sum = 0 #No heating buffer through the rooms as a good radiator model is not integrated yet
        return {"phi" : phi_sum,}
    
class Wall(Component):
    """
    Define and calculate heatflow through a wall using euler forward algorithm
    M denotes the heat transfer matrix
    WARNING: Doesn't check for euler forward stability
    
    definition
        layers : Materials and respective thicknesses of the wall (From inside out)
        area : Area of the wall
        steps : Gridpoints per layer of the wall
        interval : timestep to run
        T_in : "inside" temperature 
        T_out : "outside" temperature
        U_in : Heat contact conductivity from the air in the room to the wall [m*m*K/W]
        U_out : Heat contact conductivity the Wall into the Air outside [m*m*K/W]
    in
        t_in inside temperature in K
        t_out outside temperature in K
    out 
        phi thermal losses in W
        
        
    TODO:
    Add heat input through solar radiation absorption and radiative heat transfer
    
    """
    @classmethod
    def reg(cls, building, layers,area, ):
        wall = cls(layers, area, T_in = building.thermal_zone.T_in, T_out=building.thermal_zone.T_out)
        
        super().__init__(wall, building, connector_out = ['phi'], )
        wall.building = building
        wall.equalize()
        building.thermal_zone.connection_in.append(wall)
        building.walls.append(wall)
    
    def __init__(self, layers,area, steps = 5, interval=100, T_in=294, T_out=278 ,U_in = 1/.13, U_out=1/.04, name = None):
        """
        Initialize Values
        Build Transmission Matrix
        """
        self.T = np.ones(steps*len(layers)+2)*T_in
        self.C_V = np.zeros(steps*len(layers)+2)
        self.C_V[0] = 99
        self.C_V[-1] = 99
        self.U_ = np.zeros(steps*len(layers)*2+2)
        #self.U_[0] = 1/.25 #for reduced circ
        self.U_[0] = U_in #for free circ
        self.U_[-1] = U_out
        
        self.steps = steps #ste
        self.timestep = interval
        
        self.T_in = T_in
        self.T_out = T_out
    
        self.layers= layers
        self.area = area
        self.building = None
        self.PVGIS = None
        
        for ind, layer in enumerate(self.layers):
            for step in range(self.steps):
                self.C_V[ind*steps+step+1] = layer['c_v']*layer['thickness']*layer['rho']/self.steps
                self.U_[2*(ind*steps+step)+1] = layer["lambda"]/layer['thickness']*self.steps*2
                self.U_[2*(ind*steps+step)+2] = layer["lambda"]/layer['thickness']*self.steps*2
        ## U already accounts for the time interval!!!
        self.U = self.timestep/(1/self.U_[1::2]+1/self.U_[0::2])
        self.diagonal = self.C_V/self.C_V
        self.diagonal[1:]-=self.U/self.C_V[1:]
        self.diagonal[:-1]-=self.U/self.C_V[:-1]
        self.M = diags([self.diagonal,self.U/self.C_V[1:],self.U/self.C_V[:-1]], [0,-1,1])
        self.phi = 0
    @property
    def U_value(self):
        return {"component" : 1/np.sum(1/self.U[1:-1]), 
                "component and contact resistance" : 1/np.sum(1/self.U)}
    def step(self, T_out=None):
        """
        Update Temperatures with new boundary conditions
        """
        if T_out is None:
            T_out=self.T_out
        T_step = self.M@self.T
        
        self.T  = T_step
        self.T[0] = self.T_in
        self.T[-1] = T_out
        
    def equalize(self, T_in = 294, T_out = 283, timestep = None):
        """
        Calculate Temperature Equilibrium
        """
        if timestep is None:
            timestep = round(3600*24*3/self.timestep)
        
        self.T_in = 294
        for x in range(timestep):
            self.step(T_out=T_out)
        self.phi = self.loss()
    def run(self, T_in = None, T_out = None, timestep = 10*60):
        """
        Sum losses up while running the time forward
        return the average thermal loss in W
        Negative sign means Energy loss
        """
        if (timestep < self.timestep) or (timestep%self.timestep !=0 ):
            raise ValueError('timestep needs to be integer multiples of the time interval in the heat equation')
        steps = round(timestep/self.timestep)
        losses = []
        if T_in is None:
            try:
                T_in = self.building.thermal_zone.T_in
            except:
                T_in = 294
        if T_out is None:
            try:
                T_out = self.building.thermal_zone.T_out
            except:
                T_out = 283
            
        self.T_in = T_in
        self.T_out = T_out
        
        for x in range(steps):
            self.step(T_out=T_out)
            losses.append(self.loss())
        self.phi = -np.average(losses)
        return {'name' : self.name ,
                'phi' : self.phi}
            
    #@property
    def loss(self, positions=[1]):
        losses = []
        gains = []
        for pos in positions:
            losses.append((self.T[pos]-self.T[pos+1])*self.U[pos]*self.area/self.timestep)
        return np.average(losses)


class underfloor_radiator():
    """
    Heat distribution device
    """
    def __init__(self):
        self.area = 0
        self. layers = [{
  'rho': 2000.0,
  'c_v': 1000.0,
  'lambda': 1.2,
  'thickness': 0.04}]
        self.c_heat = layers[0]['rho']*layers[0]['thickness'] # J/(m²K)
        self.Wall = Wall(self.layers,1,10,60, zero_Celsius+30, zero_Celsius+21)
        self.Wall.equalize
    def qdot(self, T_surf, T_room):
        """
        Calculating energy transfer W/(m²K) qdot from Din EN 1264 part 2 and 3
        """
        return 8.92*(T_surf-T_room)**1.1

    
class Window(Component):
    """
    Account for thermal in/out due to windows in the building
    Phi_In is calculated through the power input from PVGIS
    Phi_Out is calculated through the thermal "wall" characteristics 
    of the window.
    
    self.Area Area in m*m
    self.az azimuth in degrees 
    self.ele elevation in degrees 
    
    in
    
    out
        phi, P_sun@(azimuth, elev)*area_window*self.transparency
    
    """
    @classmethod
    def reg(cls, building ,area, azimuth, inclination = 90, occlusion = .9, layers = None ,name = None):
        window = cls(area, azimuth)
        Wall.reg(building, window.layers, window.area, )
        window.PVGIS = PVGIS(building.location, inclination, azimuth)
        window.building = building
        building.windows.append(window)
        super().__init__(window, building, connector_out = ['phi'], )
        building.thermal_zone.connection_in.append(window)
    
    def __init__(self, area, az, ele = 90, occlusion = .9, layers = None, name = None):
        self.area = area
        self.az = az
        self.ele = ele
        self.occlusion = occlusion
        #self.normal = az_el2norm(self.az,self.ele)
        if layers is None:
            self.layers = DEFAULT_LAYERS['Window']
        else:
            self.layers =layers
        self.PVGIS = None
        self.building = None
        self.phi = 0
    def run(self):
        """
        Return thermal power ingress in W
        """
        phi_in_m2 = self.PVGIS[self.building.date][['Bi', 'Di', 'As']].sum()
        self.phi = phi_in_m2 * self.area * self.occlusion
        return {'name' : self.name ,
                'phi' : self.phi}


class Heatpump(Component):
    """
    Heatpump device including thermal water buffer for
    heat and warmwater
    self.c_heat = 500* 4300 # kg*J/kg/K
    self.c_warmwater = 500* 4300 # kg*J/kg/K
    
    self.T_heat = [303, 300, 310] #curr, min, max K
    self.T_warmwater = [313, 310, 318] #curr, min, max K
    
    self.T_eq_in = 9+zero_Celsius #K 
    self.T_eq_out = 4+zero_Celsius #K
    self.P_th_max= 23e3 # 
    self.P_e_max = 8.5e3 # 
    self.P_e_p = .5
    
    self.P_th = self.P_th_max
    self.state = None 
    self.need = None
    self.P_el_in = 0
    """
    @classmethod
    def reg(cls, building):
        heatpump = cls()
        super().__init__(heatpump, building, connector_in = ['phi','phi_ww'], connector_out = ['p_in','T_heat'], connection_in = [building.thermal_zone])
        building.components.update({heatpump.name :heatpump})
        heatpump.timestep = building.timestep
    def __init__(self):
        self.c_heat = 500* 4300 # kg*J/kg/K
        self.c_warmwater = 500* 4300 # kg*J/kg/K
        
        self.T_heat = [301, 300, 310] #curr, min, max K
        self.T_warmwater = [313, 310, 318] #curr, min, max K
        
        self.T_eq_in = 9+zero_Celsius #K 
        self.T_eq_out = 4+zero_Celsius #K
        self.P_th_max= 23e3 #Maximum thermal output power
        self.P_e_max = 8.5e3 #Maximum electrical input power
        self.P_e_p = .4 #Proportional power factor
        
        self.P_th = self.P_th_max
        self.state = None 
        self.need = None
        self.P_el = 0
        
        self.timestep=600
    
    def run(self, phi, phi_ww=0):
        """
        Update values based on thermal losses and programming settings
        negative phi means energy is going out
        """
        
        interval = self.timestep
        self.T_heat[1] = self.T_heat[1] + phi*interval/self.c_heat
        self.T_warmwater[1] = self.T_warmwater[1] + phi_ww*interval/self.c_warmwater
        return self.go()
        
    def P_K(self, T_target):
        """
        Calculate thermal power dependent on target temperature
        """
        P_th = (14e3+4e3* ((55+273-T_target)/(25)))*self.P_e_p/.5
        #print(P_th)
        return P_th
        
    
    def go(self):
        """
        Heat if needed, Prioritize WW
        """
        interval = self.timestep
        if self.T_heat[1] < self.T_heat[0]:
            self.state = "Heat"
        elif (self.state == "Heat") and (self.T_heat[1] < self.T_heat[2]):
            self.state = "Heat"
        elif self.T_heat[1] > self.T_heat[2]:
            self.state = None
        elif self.T_warmwater[1] < self.T_warmwater[0]:
            self.state = "WW"
        elif self.T_warmwater[1] > self.T_warmwater[2]:
            self.state = None
        else:
            pass
            #self.state=None
        if self.state:
            self.P_el = self.P_e_max*self.P_e_p
        else:
            self.P_el = 0
        if self.state == "WW":
            self.T_warmwater[1] = self.T_warmwater[1]+self.P_K(T_target=self.T_warmwater[2])*interval/self.c_heat
        elif self.state == "Heat":
            self.T_heat[1] = self.T_heat[1] + self.P_K(T_target=self.T_heat[2])*interval/self.c_heat
        else:
            pass
        return {"p_in" : -self.P_el,
               "name" : self.name,
                "T_heat" : self.T_heat[1],
                "T_WW" : self.T_warmwater[1],
               }

#layers = gen_layers([find_material('Ziegel, 1800'), find_material('granit')],[.3,.4])
