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


class module():
    def __init__(self,i_sc=6.48, v_oc=69.5/96, i_mp=6.09, v_mp=59.1/96):
        self.i_sc = i_sc
        self.v_oc = v_oc
        self.i_mp = i_mp
        self.v_mp = v_mp
        self.v_rb = -0.5
        
        self.v_i = single2ifromv()
        self.v_i_P = None
    def mod_config(self):
        return np.asarray([self.i_sc, self.v_oc, self.i_mp, self.v_mp])
        
        
def get_pv_fit_func(i_sc=6.48, v_oc=69.5/96, i_mp=6.09, v_mp=59.1/96):
    """
    Returns fit function for pv-cell parameters using datasheet information

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
    def pv_fit_func(I_ph, I_0, R_s, R_sh, v_th):
        pv = pvsystem.singlediode(I_ph, I_0, R_s, R_sh, v_th)
        return np.sum(np.abs(np.asarray([pv['i_sc']-i_sc, pv['v_oc']-v_oc,pv['i_mp']-i_mp ,pv['v_mp']-v_mp ])) , axis = 0)
def fit_pv(i_sc=6.48, v_oc=69.5/96, i_mp=6.09, v_mp=59.1/96, Ns=4, x0=None):
    """
    Given

    i_sc  short circuit current
    v_oc  open circuit voltage
    i_mp  mpp current
    v_mpp  mpp voltage

    Ns number of iteration steps

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
    ranges = np.asarray([x_0-x_0*.8, x_0+x_0*.8])
    fit_func = get_pv_fit_func(i_sc=i_sc, v_oc=v_oc, i_mp=i_mp, v_mp=v_mp)
    pv_param = brute(fit_pvcurve, ranges=tuple(zip(ranges[0],ranges[1])), Ns=Ns)
    return pv_param



def single2ifromv_(arg0 = np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02])):
    (I_ph, I_0, R_s, R_sh, v_th_) = arg0
    #@functools.lru_cache(maxsize=2048*16)
    def v_from_i(I, L,t_cell):
        t = pvlib.pvsystem.v_from_i(R_sh, R_s, v_th_*(t_cell+273)/298.5, np.asarray(I), I_0, I_ph*np.asarray(L))  
        return t
    return v_from_i

def single2ifromv(arg0 = np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02])):
    (I_ph, I_0, R_s, R_sh, v_th_) = arg0
    @functools.lru_cache(maxsize=2048*16)
    def v_from_i(I, L,t_cell):
        t = pvlib.pvsystem.v_from_i(R_sh, R_s, v_th_*(t_cell+273)/298.5, np.asarray(I), I_0, I_ph*np.asarray(L))
        if np.isnan(t).any():
            return v_i__((I,L,t_cell))
        else:
            return t
    return v_from_i


def single2ifromv_P(arg0 = np.array([6.48000332e+00, 6.37762333e-10, 8.45318984e-04, 1.65194938e+03,
       3.14194723e-02])):
    (I_ph, I_0, R_s, R_sh, v_th_) = arg0
    def v_from_i(I, L,t_cell):
        return pvlib.pvsystem.v_from_i(R_sh, R_s, v_th_*(t_cell+273)/298.5, np.asarray(I), I_0, I_ph*np.asarray(L))
    return v_from_i

v_i_P = single2ifromv_P()

with futures.ProcessPoolExecutor() as e:
    @functools.lru_cache(maxsize=2048)
    def v_i_P(*args, **kwargs):
        (I, L,v_th) = args
        return e.submit(v_i_P_(np.asarray(I)[:,None],np.asarray(L)[None,:],  v_th,**kwargs), *args, **kwargs)


@functools.lru_cache(maxsize=2048)
def substr_v(i, L, v_th=0, v_rb=-.5,):
    """
    Returns voltages of substring at current I
    """
    v = np.maximum(np.sum(v_i(float(i),tuple(L),  v_th )), v_rb*np.exp(i/20))
    return v

@functools.lru_cache(maxsize=2048*16)
def substr_v_P(i, L, t_cell=0, v_rb=-.5):
    """
    Returns voltages of substring at current i given illumination L
    """

    return np.maximum(np.sum(v_i_P(i,L,  t_cell ), axis=1),  v_rb*np.exp(np.asarray(i)/20))

#@functools.lru_cache(maxsize=2048)
def calc_t_cell(L_, T_am, W_10, model='roof_mount_cell_glassback'):
    """
    Wrapper function for cell temperature calculation
    """
    return pvsystem.sapm_celltemp(np.sum(np.hstack(L_))/np.size(np.hstack(L_))*1e3, W_10, T_am,  )['temp_cell'][0]


def V_module(i,L_,T_am, W_10, rb=-.5):
    """
    return voltage of array at Illumination L[]
    """
    t_cell =calc_t_cell(L_, T_am, W_10)
    return np.asarray([sum(substr_v(i_, L=tuple(L)) for L in L_) for i_ in i])



def V_mod_P(args):
    (i,L_,T_am, W_10, rb) = args
    """
    return voltage of array at Illumination L[]
    """
    t_cell =calc_t_cell(L_, T_am, W_10)
    return np.asarray(sum(substr_v_P(tuple(i), L=tuple(L), t_cell=t_cell) for L in L_))



#v_i_no_cache = single2ifromv_()
#I = np.linspace(-4,7,300)
#L = np.linspace(0,1.2,100)
#T = np.linspace(-20,80,100)
#data = v_i_no_cache(*np.meshgrid(I, L, T, indexing='ij', sparse=True))
#v_i__ = RegularGridInterpolator((I,L,T), data, fill_va)
#data = v_i_no_cache(*np.meshgrid(I, L, T))
#I_ = np.meshgrid(I, L, T)[0].flatten()
#L_ = np.meshgrid(I, L, T)[1].flatten()
#T_ = np.meshgrid(I, L, T)[2].flatten()
#not_nans = np.argwhere(np.logical_not(np.isnan(data.flatten()))).reshape(-1)
#v_i__ = NearestNDInterpolator((I_.flatten()[not_nans], L_.flatten()[not_nans], T_.flatten()[not_nans]), data.flatten()[not_nans])