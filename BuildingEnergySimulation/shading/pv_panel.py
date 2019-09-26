#def single2ifromv(arg0 = np.array([6.48000237e+00, 6.03959251e-10, 5.55129794e-03, 1.52143849e+04, 3.13453068e-02])):

import numpy as np

from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import brute
import functools
import pvlib
from pvlib import pvsystem
from concurrent import futures
"""
One Class for Module:
Fit PV-Parameters from Datasheet
single_ifromv (singlediode)
substri_v (substring)
module_v (module)
"""

class Pv_panel():
    """
        i_sc = i_sc short circuit current in A
        v_oc = v_oc open circuit current in V
        i_mp = i_mp Current at maximum power point stp
        v_mp = v_mp Volrtage at maximum power point stp
        n_cells = 96 Number of cells in panel
        bypass_config = [[2,4,2], 12] configuration of the bypass diodes. I this
                        case the panel consists of 8 columns and 12 rows of 96
                        cells. The row 1-2, 3-6, 7-8 each have their diode.
        dim_x = 1.046 width of the panel in meters
        dim_y = 1.558 height of the panel in meters
        v_rb = -0.5 Reverse bias diode breakthrough voltage in V
        v_i = single2ifromv() get the V-I-function for a single diode
        I_max = 7 maximum current in the Simulation
        I_min = -.4 minimum current in the Simulation
    """

    def __init__(self,i_sc=6.48, v_oc=69.5, i_mp=6.09, v_mp=59.1 ,n_cells=96,
                bypass_config = [[2,4,2], 12], dim_x = 1.046, dim_y = 1.558,
                v_rb = -0.5, I_min = -4, I_max = 7, L_max = 1.2 ):
        self.i_sc = i_sc
        self.v_oc = v_oc
        self.i_mp = i_mp
        self.v_mp = v_mp
        self.n_cells = n_cells
        self.bypass_config = bypass_config
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.cells_x = np.sum(bypass_config[0])
        self.cells_y = np.sum(bypass_config[1])
        self.cell_dim_x = self.dim_x/self.cells_x
        self.cell_dim_y= self.dim_y/self.cells_y
        self.v_rb = v_rb
        self.cell_param = None
        self.v_from_i = None
        self.I_min = I_min
        self.I_max = I_max
        self.L_max= L_max
        self.fit_cell_parameters()
        self.set_v_from_i()
    def fit_cell_parameters(self, cell_config=None):
        if cell_config is None:
            self.cell_param = fit_pv(**self.cell_config())
        else:
            self.cell_param = fit_pv(**cell_config)
    def set_v_from_i(self):
        self.v_from_i = get_v_panel_from_i(cell_param = self.cell_param,
                I_max =self.I_max,
                I_min =self.I_min,
                L_max =self.L_max,
        )
    def cell_config(self):
        return {
        'i_sc': self.i_sc,
        'v_oc': self.v_oc/self.n_cells,
        'i_mp': self.i_mp,
        'v_mp': self.v_mp/self.n_cells
        }


def fit_pv(i_sc=6.48, v_oc=69.5/96, i_mp=6.09, v_mp=59.1/96, Ns=4, x0=None, delta_x = .8):
    """
    Given

    i_sc  short circuit current
    v_oc  open circuit voltage
    i_mp  mpp current
    v_mpp  mpp voltage

    Ns = Number of increments for brute force range
    x0 = Starting values for fit
    delta_x = relative deviation from x0 to explore



    the function brute forces the best fit for

    I_ph photo current
    I_0 reverse saturation current
    R_s series resistance
    R_sh shunt resistance
    v_th thermal voltage kT/e
    """
    if x0 == None:
        x0 = (6.09, 4.268694026502061e-10, .0045, 10000, 0.025679644404907293)
    x_0 = np.asarray(x0)
    ranges = np.asarray([x_0-x_0*delta_x, x_0+x_0*delta_x])
    pv_loss_func = gen_pv_fit_loss_func(i_sc=i_sc, v_oc=v_oc, i_mp=i_mp, v_mp=v_mp)
    pv_param = brute(pv_loss_func, ranges=tuple(zip(ranges[0],ranges[1])), Ns=Ns)
    return pv_param


def gen_pv_fit_loss_func(i_sc=6.48, v_oc=69.5/96, i_mp=6.09, v_mp=59.1/96):
    """
    Returns loss function for physical pv-cell parameters using datasheet information

    i_sc  short circuit current
    v_oc  open circuit voltage
    i_mp  mpp current
    v_mpp  mpp voltage

    Fits

    I_ph photo current
    I_0 reverse saturation current
    R_s series resistance
    R_sh shunt resistance
    v_th thermal voltage kT/e
    """
    def pv_fit_loss_func(params):
        """
        Loss function for physical pv-cell parameters using datasheet information
        """
        I_ph, I_0, R_s, R_sh, v_th = params
        pv = pvsystem.singlediode(I_ph, I_0, R_s, R_sh, v_th)
        return np.sum(np.abs(np.asarray([pv['i_sc']-i_sc, pv['v_oc']-v_oc,pv['i_mp']-i_mp ,pv['v_mp']-v_mp ])) , axis = 0)
    return pv_fit_loss_func



def get_v_panel_from_i(cell_param=np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02]), I_min = -4, I_max = 7, L_max = 1.2):
    """
    Returns a function to return the voltages of a pv-panel including reverse bias diode given the
    physical cell parameters (I_ph, I_0, R_s, R_sh, v_th) and the arguments
        I : current through the cell in A
        L : Photocurrent in A (Is considered proportional to the irradiance)
        t_cell : Cell temperature in Celsius
        v_rb : bypass diode breakthrough voltage in V (reverse bias)

    Parameters:
    cell_param : tuplelike(
                            I_ph, Photocurrent
                            I_0, Darkcurrent
                            R_s, Series resistance
                            R_sh, Shunt resistance
                            v_th thermal voltage kT/e
                            )
    """


    def single2v_from_i_with_nan(arg0 = np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02])):
        (I_ph, I_0, R_s, R_sh, v_th) = arg0
        """
        Return volts from i function for single pv-cell (diode) without the nan catch
        """
        def v_from_i(I, L,t_cell):
            """
            Return diode voltage for a  single pv-cell (diode) given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
            and the arguments
            I : current through the cell in A
            L : Photocurrent in A (Is considered proportional to the irradiance)
            t_cell : Cell temperature
            """
            v_pn= pvlib.pvsystem.v_from_i(R_sh, R_s, v_th*(t_cell+273)/298.5, np.array(I, ndmin=2).T, I_0, I_ph*np.asarray(L))
            return v_pn
        return v_from_i

    """
    Generate interpolation function to guarantee non nan values
    """
    v_i_with_nan = single2v_from_i_with_nan(arg0 = cell_param)
    I = np.linspace(I_min,I_max,110)
    L = np.linspace(0,L_max,100)
    T = np.linspace(-20,80,100)
    data = v_i_with_nan(*np.meshgrid(I, L, T))
    I_ = np.meshgrid(I, L, T)[0].flatten()
    L_ = np.meshgrid(I, L, T)[1].flatten()
    T_ = np.meshgrid(I, L, T)[2].flatten()
    not_nans = np.argwhere(np.logical_not(np.isnan(data.flatten()))).reshape(-1)
    v_i_interpolate = NearestNDInterpolator((I_.flatten()[not_nans], L_.flatten()[not_nans], T_.flatten()[not_nans]), data.flatten()[not_nans])


    def single2v_from_i(arg0 = np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02])):
        """
        Return v from i function for single pv-cell (diode) with nan catch through linear interpolation
        """
        (I_ph, I_0, R_s, R_sh, v_th) = arg0
        @functools.lru_cache(maxsize=2048*16)
        def v_from_i(I, L,t_cell):
            """
            Return diode voltage for a  single pv-cell (diode) given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
            and the arguments
            I : current through the cell in A
            L : Photocurrent in A (Is considered proportional to the irradiance)
            t_cell : Cell temperature
            """
            v_pn = pvlib.pvsystem.v_from_i(R_sh, R_s, v_th*(t_cell+273)/298.5, np.array(I, ndmin=2).T, I_0, I_ph*np.asarray(L))
            if np.isnan(v_pn).any():
                return v_i_interpolate((np.array(I, ndmin=2).T,np.asarray(L),t_cell))
            else:
                return v_pn
        return v_from_i
    v_from_i = single2v_from_i(cell_param)

    def calc_t_cell(L_, T_am, W_10, model='roof_mount_cell_glassback'):
        """
        Wrapper function for cell temperature calculation
        """
        return pvsystem.sapm_celltemp(np.sum(np.hstack(L_))/np.size(np.hstack(L_))*1e3, W_10, T_am,  )['temp_cell'][0]

    @functools.lru_cache(maxsize=2048*16)
    def substr_v_P(i, L, t_cell=0, v_rb=-.5):
        """
        Returns voltages of a substring including reverse bias diode given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
        and the arguments
        I : current through the cell in A
        L : Photocurrent tuple for all cells in substring in A (Is considered proportional to the irradiance)
        t_cell : Cell temperature in Celsius
        v_rb : bypass diode breakthrough voltage in V (reverse bias)
        """

        return np.maximum(np.sum(v_from_i(i,L,  t_cell ), axis=1),  v_rb*np.exp(np.asarray(i)/20))

    def v_from_i_panel(args):
        """
        Returns voltages of a pv-panel including reverse bias diode given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
        and the arguments
        I : current through the cell in A
        L : Photocurrent in A (Is considered proportional to the irradiance)
        t_amb : Cell temperature in Celsius
        W_10 : windspeed in 10m
        """
        (i,L,T_am, W_10, rb) = args
        t_cell =calc_t_cell(L, T_am, W_10)
        return np.asarray(sum(substr_v_P(tuple(i), L=tuple(L_substring), t_cell=t_cell) for L_substring in L))
    return v_from_i_panel
