"""Parametrizing a single pv-panel.

Fit PV-Parameters from Datasheet
single_ifromv (singlediode)
substri_v (substring/module)
module_v (module/panel)
"""
#
# def single2ifromv(arg0 = np.array(
# [6.48000237e+00, 6.03959251e-10, 5.55129794e-03,
# 1.52143849e+04, 3.13453068e-02])):


import numpy as np

from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import brute
from typing import List, Dict, Callable
import functools
import pvlib
from pvlib import pvsystem
from concurrent import futures



class Pv_panel():
    """Class to parametrize a single pv-panel.

    Attributes:
        cells_x (int): Number of cells along x-direction.
        cells_y (int): Number of cells along y-direction.
        cell_dim_x (float): cell size in x-direction in meters.
        cell_dim_y (float): cells size in y direciton in meters.
        cell_param (dict): (approximated) physical cell paremeters.
        v_from_i (fun): Panel voltage-current-function.

    """

    def __init__(self, i_sc: float = 6.48, v_oc: float = 69.5,
                 i_mp: float = 6.09, v_mp: float = 59.1, n_cells: int = 96,
                 bypass_config: list = [[2, 4, 2], 12], dim_x: float = 1.046,
                 dim_y: float = 1.558, v_rb: float = -0.5, I_min: float = -4,
                 I_max: float = 7, L_max: float = 1.2):
        """Initialize.

        Args:
            i_sc (float): i_sc panel short circuit current in A.
                Defaults to 6.48.
            v_oc (float): v_oc panel open circuit current in V.
                Defaults to 69.5.
            i_mp (float): i_mp panel current at maximum power point stp.
                Defaults to 6.09.
            v_mp (float): v_mp panel voltage at maximum power point stp.
                Defaults to 59.1.
            n_cells (int): Number of cells in panel. Defaults to 96.
            bypass_config (list): configuration of the bypass diodes. I this
                case the panel consists of 8 columns and 12 rows of 96
                cells. The row 1-2, 3-6, 7-8 each have their diode.
                Defaults to [[2, 4, 2], 12].
            dim_x (float): Width of the panel in meters. Defaults to 1.046.
            dim_y (float): Height of the panel in meters. Defaults to 1.558.
            v_rb (float): Reverse bias diode breakthrough voltage in V.
                Defaults to -0.5.
            I_min (float): Minimum current in the calculation.
                Defaults to -4 A.
            I_max (float): Maximum current in the calculation. Defaults to 7 A.
            L_max (float): Maximum Irradiance. Defaults to 1.2 kw/m*m.
        """
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
        self.cell_dim_y = self.dim_y/self.cells_y
        self.v_rb = v_rb
        self.cell_param = None
        self.v_from_i = None
        self.I_min = I_min
        self.I_max = I_max
        self.L_max = L_max
        self.fit_cell_parameters()
        self.set_v_from_i()

    def fit_cell_parameters(self, cell_config: dict = None):
        """Fit and set physical cell paremeters.

        Args:
            cell_config (dict): Dictionary of panel parameters. If nothing is
                provided, it takes the values from the instance creation
                Defaults to None.

        """
        if cell_config is None:
            self.cell_param = fit_pv(**self.cell_config())
        else:
            self.cell_param = fit_pv(**cell_config)

    def set_v_from_i(self):
        """Set voltage-current function."""
        self.v_from_i = get_v_panel_from_i(cell_param=self.cell_param,
                                           I_max=self.I_max,
                                           I_min=self.I_min,
                                           L_max=self.L_max,
                                           )

    def cell_config(self) -> Dict:
        """Return cell config as dictionary."""
        return {
            'i_sc': self.i_sc,
            'v_oc': self.v_oc/self.n_cells,
            'i_mp': self.i_mp,
            'v_mp': self.v_mp/self.n_cells
        }


def fit_pv(i_sc: float = 6.48, v_oc: float = 69.5/96, i_mp: float = 6.09,
           v_mp: float = 59.1/96, Ns: int = 4, x0=None, delta_x: float = .8):
    """Brute force approximation of physical cell parameters.

    the function brute forces the best fit for the phyiscal cell parameters of
    the singlediode model and returns a dictionary with the values
     - I_ph photo current
     - I_0 reverse saturation current
     - R_s series resistance
     - R_sh shunt resistance
     - v_th thermal voltage kT/e

    Args:
        i_sc (float): i_sc cell short circuit current in A. Defaults to 6.48.
        v_oc (float): v_oc cell open circuit current in V. Defaults to 69.5/96.
        i_mp (float): i_mp cell current at maximum power point stp.
            Defaults to 6.09.
        v_mp (float): v_mp cell voltage at maximum power point stp.
            Defaults to 59.1/96.
        Ns (type): Number of steps per dimension for brute force parameter
            space. Defaults to 4.
        x0 (type): Origin of the parameter space for the brute force
            optimization. Defaults to None.
        delta_x (type): Maximum deviation (extension) of the brute force grid.
            Defaults to .8.

    Returns:
        pv_param (dict): Physical cell parameters

    """
    if x0 is None:
        x0 = (6.09, 4.268694026502061e-10, .0045, 10000, 0.025679644404907293)
    x_0 = np.asarray(x0)
    ranges = np.asarray([x_0-x_0*delta_x, x_0+x_0*delta_x])
    pv_loss_func = _gen_pv_fit_loss_func(
        i_sc=i_sc, v_oc=v_oc, i_mp=i_mp, v_mp=v_mp)
    pv_param = brute(pv_loss_func, ranges=tuple(
        zip(ranges[0], ranges[1])), Ns=Ns)
    return pv_param


def _gen_pv_fit_loss_func(i_sc: float = 6.48, v_oc: float = 69.5/96,
                          i_mp: float = 6.09,
                          v_mp: float = 59.1/96) -> Callable:
    """Return a loss function for physical pv-cell parameters."""
    def pv_fit_loss_func(params):
        """Loss function for physical pv-cell parameters."""
        I_ph, I_0, R_s, R_sh, v_th = params
        pv = pvsystem.singlediode(I_ph, I_0, R_s, R_sh, v_th)
        return np.sum(np.abs(np.asarray([pv['i_sc']-i_sc, pv['v_oc']-v_oc,
                                         pv['i_mp']-i_mp, pv['v_mp']-v_mp])),
                      axis=0)
    return pv_fit_loss_func


def get_v_panel_from_i(cell_param=np.array([6.48000332e+00, 6.37762333e-10,
                                            8.45318984e-04, 1.65194938e+03,
                                            3.14194723e-02]),
                       I_min: float = -4, I_max: float = 7,
                       L_max: float = 1.2) -> Callable:
    """Returns a function to calculate voltages of a pv-panel at given Current.

    Includes reverse bias diode.
    Requiers physical cell parameters.
     - I_ph photo current
     - I_0 reverse saturation current
     - R_s series resistance
     - R_sh shunt resistance
     - v_th thermal voltage kT/e

    Args:
        cell_param (tuplelike): Physical Cell Parameters.
            Defaults to np.array([6.48000332e+00, 6.37762333e-10,
            8.45318984e-04, 1.65194938e+03, 3.14194723e-02]).
        I_min (type): Minimum current to be considered. Defaults to -4.
        I_max (type): Maximum current to be considered. Defaults to 7.
        L_max (type): Maximum Photocurrent to be considered in A.
            Defaults to 1.2.

    """

    def single2v_from_i_with_nan(arg0=np.array([6.48000332e+00, 6.37762333e-10,
                                                8.45318984e-04, 1.65194938e+03,
                                                3.14194723e-02])) -> Callable:
        """Return function to calculate voltage from curent of a single diode.

        Might include Nan.

        Args:
            arg0 (type): . Defaults to np.array(
                [6.48000332e+00, 6.37762333e-10,
                8.45318984e-04, 1.65194938e+03, 3.14194723e-02]).

        Returns:
            Callable: Calculate voltage from current of a single diode.

        """
        (I_ph, I_0, R_s, R_sh, v_th) = arg0

        def v_from_i(I: np.ndarray, L: np.ndarray, t_cell: float):
            """Return diode voltage for a single pv-cell (diode).

            Given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
            and the arguments

            Args:
                I (np.ndarray): Current through the cell in A.
                L (np.ndarray): Photocurrent in A .
                    (Is considered proportional to the irradiance)
                t_cell (float): Cell temperature.

            Returns:
                np.ndarray: Voltage at given current without NAN catch.

            """

            v_pn = pvlib.pvsystem.v_from_i(
                R_sh, R_s, v_th*(t_cell+273)/298.5, np.array(I, ndmin=2).T,
                I_0, I_ph*np.asarray(L))
            return v_pn
        return v_from_i

    # Generate interpolation function to guarantee non nan values

    v_i_with_nan = single2v_from_i_with_nan(arg0=cell_param)
    I_arr = np.linspace(I_min, I_max, 110)
    L_arr = np.linspace(0, L_max, 100)
    T_arr = np.linspace(-20, 80, 100)
    data = v_i_with_nan(*np.meshgrid(I_arr, L_arr, T_arr))
    I_ = np.meshgrid(I_arr, L_arr, T_arr)[0].flatten()
    L_ = np.meshgrid(I_arr, L_arr, T_arr)[1].flatten()
    T_ = np.meshgrid(I_arr, L_arr, T_arr)[2].flatten()
    not_nans = np.argwhere(np.logical_not(
        np.isnan(data.flatten()))).reshape(-1)
    v_i_interpolate = NearestNDInterpolator(
        (I_.flatten()[not_nans], L_.flatten()[not_nans],
         T_.flatten()[not_nans]), data.flatten()[not_nans]
        )

    def single2v_from_i(arg0=np.array([6.48000332e+00, 6.37762333e-10,
                                       8.45318984e-04, 1.65194938e+03,
                                       3.14194723e-02])) -> Callable:
        """Return function to calculate voltage from curent of a single diode.

        Includes NAN catch.

        Args:
            arg0 (np.ndarray): Physical cell parameters
                (I_ph, I_0, R_s, R_sh, v_th).
                Defaults to np.array([6.48000332e+00, 6.37762333e-10,
                8.45318984e-04, 1.65194938e+03, 3.14194723e-02]).

        Returns:
            Voltage at given current with NAN catch

        """

        (I_ph, I_0, R_s, R_sh, v_th) = arg0
        pvlib_v_from_i = pvlib.pvsystem.v_from_i
        @functools.lru_cache(maxsize=2048*16)
        def v_from_i(I_cells, Iph, t_cell):
            """Return diode voltage for a single pv-cell (diode).

            Given the physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
            and the arguments

            Includes NAN catch.

            Args:
                I_cells (tuple): Current through the cell in A.
                Iph (tuple): Photocurrent in A .
                    (Is considered proportional to the irradiance)
                t_cell (float): Cell temperature.

            Returns:
                np.ndarray: Voltage at given current with NAN catch.

            """
            v_pn = pvlib_v_from_i(
                R_sh, R_s, v_th*(t_cell+273)/298.5,
                np.array(I_cells, ndmin=2).T,
                I_0, I_ph*np.asarray(Iph))
            if np.isnan(v_pn).any():
                return v_i_interpolate((np.array(I_cells, ndmin=2).T,
                                        np.asarray(Iph), t_cell))
            else:
                return v_pn
        return v_from_i
    v_from_i = single2v_from_i(cell_param)

    def calc_t_cell(L: np.ndarray, T_am: float, W_10: float,
                    model: str = 'roof_mount_cell_glassback'):
        """Wrapper function for cell temperature calculation
        Args:
            L (np.ndarray): Irradiance in kw.
            T_am (float): Ambient temperature.
            W_10 (float): Windspeed @10 meter.
            model (str): Defaults to 'roof_mount_cell_glassback'.

        Returns:
            float: Cell temperature in Kelvin.

        """
        return pvsystem.sapm_celltemp(
            np.sum(np.hstack(L))/np.size(np.hstack(L))*1e3,
            W_10, T_am,)['temp_cell'][0]

    @functools.lru_cache(maxsize=2048*16)
    def substr_v_P(I_substr: tuple, Iph_substr: tuple, t_cell: float = 0,
                   v_rb: float = -.5):
        """Returns voltages of a substring in a panel at given currents.

        Returns voltages of a pv-panel including reverse bias diode given the
            physical cell parameters (I_ph, I_0, R_s, R_sh, v_th)
            and the arguments

        Args:
            I_substr (tuple): Current through the cell in A.
            Iph_substr (tuple): Photocurrent in A .
            t_cell (float, optional): Cell temperature. Defaults to 0.
            v_rb (float, optional): bypass diode breakthrough voltage in V
                (reverse bias). Defaults to -.5.

        Returns:
            np.ndarray: Voltages at given currents through the substring.
        """

        return np.maximum(np.sum(v_from_i(I_substr, Iph_substr,
                                          t_cell), axis=1),
                          v_rb*np.exp(np.asarray(I_substr)/20))

    def v_from_i_panel(args):
        """Returns voltages of a pv-panel at given currents.

        Args:
            args (tuple): (
                I_pan : current through the cell in A
                Iph_panel : List of Photocurrents in A
                    (Is considered proportional to the irradiance)
                t_amb : Cell temperature in Celsius
                W_10 : windspeed in 10m
            )
        """
        (I_pan, Iph_panel, T_am, W_10, _) = args
        t_cell = calc_t_cell(Iph_panel, T_am, W_10)
        return np.asarray(
            sum(substr_v_P(tuple(I_pan),
                           Iph_substr=tuple(Iph_substring),
                           t_cell=t_cell) for Iph_substring in Iph_panel)
            )
    return v_from_i_panel
