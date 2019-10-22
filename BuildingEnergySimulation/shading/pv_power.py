"""Main functionality to calculate array-I-V-curves.

Every relevant paramter is contained inside the solar_array class to run the
actual calculations.

TODO:
    Expose Illumination to photocurrent function
"""


from __future__ import annotations
import io
import os
import requests
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

from scipy.ndimage import zoom
from ..pvgis import Pvgis
from .solar import SunData
from .pv_panel import Pv_panel
# from .solar import *
if TYPE_CHECKING:
    from ..construction import Building

class Shading():
    """Class to contain manage rendered images.

    Attributes:
        images (Dict[str, np.ndarray]): Dictionary of rendered images.
        shape (tuple(int,int)): Shape of rendered images in px.
        load_images (type):
        dim_x (float): Size of image in X direction in m
        dim_y (float): Size of image in Y direction in m
        avg (np.ndarray): Non weighted average of all rendered images
            (For display purposes).
        building (Building): Root building for the Shading
        zero (np.ndarray): Zero direct irradiaton 'image'
        render_dir (str): Path to rendered images
        suns (SunData): Sun object for current location . (Includes horizon)
        scale (float): Image scale in px/m. Defaults to 20 px/m

    """

    def __init__(self, render_dir: str, suns: SunData, scale: float = 0):
        """Initialize Shading with the following arguments.

        Args:
            render_dir (str): Path to rendered images
            suns (SunData): Sun object for current location
            scale (float): Image scale in px/m. Defaults to 20 px/m

        Returns:
            Shading: Object instance

        """
        self.render_dir = Path(render_dir)
        self.images, self.shape = self.load_images(render_dir)
        self.suns = suns
        if scale is 0:
            self.scale = 20.0
        else:
            self.scale = scale
        self.dim_x = self.shape[1]/self.scale
        self.dim_y = self.shape[0]/self.scale
        self.avg = self.average_img()
        self.building = None
        self.zero = np.zeros(self.shape)

    def load_images(self, render_dir: str):
        """Load rendered images and append to dictionary.

        Args:
            render_dir (str): Directory of rendered images

        Returns:
            Dict[str, np.ndarray]: Dictionary of rendered images.
            shape (tuple(int,int)): Shape of rendered images in px.

        """
        render_path = Path(render_dir)
        image_paths = os.listdir(render_path)
        images: Dict['str', 'np.ndarray'] = {}
        for file_path in image_paths:
            if file_path[-3:] == 'png':
                images.update(self.fname_to_dict(
                    render_path.joinpath(Path(file_path))))
            else:
                print('Skipped {}. Not a png file'.format(file_path))
        return images, images[next(iter(images))].shape

    def fname_to_dict(self, file_path: Path, threshhold: float = .01):
        """Load an image from file_path and create a dict based on the
           filename.

        Args:
            file_path (Path): Path to image.
            threshhold (float): Binarisation threshhold

        Returns:
            Dict[str, np.ndarray]: .

        Todo:
            newer versions of matplotlib are compatbile with pathlib, making
            the string conversion useless

        """
        img = (mpimg.imread(str(file_path)) > threshhold).astype(np.bool)
        return({file_path.stem: img})

    def average_img(self):
        """Calculate the non weighted average of all rendered images.

        Returns:
            np.ndarray: Non weighted average

        """
        img_sum = np.zeros(self.shape)
        for ind, (_, image) in enumerate(self.images.items()):
            img_sum += image
        return img_sum/ind

    def __getitem__(self, index):
        """Return shading image corresponding to index.

        Args:
            index (List[int,int], dt.datetime): index to get the shading image.
                Can be a list containing an integer of the sun azimuth and
                elevation or a datetime. Using datetime requires a building.

        Returns:
            image np.ndarray: requested shading image as numpy array.

        """
        if isinstance(index, list):
            az, al = index
            return self.images['{}_{}'.format(az, al)]
        elif isinstance(index, dt.datetime):
            "Need to register a building to use datetime"
            assert not(
                self.building is None)
            sun = self.building.sundata[index]
            if sun['is_up']:
                az, al = int(sun['azimuth']), int(sun['elevation'])
                return self.images['{}_{}'.format(az, al)]
            else:
                return self.zero
        else:
            raise ValueError(
                'Index needs to be a list of azimuth and elevation' +
                ', or datetime')


class solar_array():
    """Main class for shaded solar array simulation.

    Attributes:
        shading (type): Shading object for direct irradiaton shadows.
        offset_x (float): Offset of the origin of the array along the x-Axis
            from the top left corner of the image. Defaults to 0.
        offset_y (float): Offset of the origin of the array along the y-Axis
            from the top left corner of the image. Defaults to 0.
        grid (type):
        timestep (int): Simulation timestep in seconds.
        building (Building): Building the array is attached to.
        pv_panel (Pv_panel): Pv_panel object, containing information about the
            used panels in the array
        azimuth (float): azimuth orientation of the array
        inclination (float): inclination of the array
        resolution_I (int): Current (A) resolution for the simulation:
            Defaults to 110.
        start_year (int): Initial year for the simulation. Defaults to 2007.
        end_year (int): Last year for the simulation. Defaults to 2008.
        output_dir (str): Path to save i-v-curves.

    """

    def __init__(self, shading: Shading, building: Building,
                 pv_panel: Pv_panel, render_dir: str, azimuth: float,
                 inclination: float, resolution_I: int = 110,
                 start_year: int = 2007, end_year: int = 2008):
        """Initialize the solar_array.

        Args:
            shading (Shading): Shading object for direct irradiaton shadows.
            building (Building): Building the array is attached to
            pv_panel (Pv_panel): Pv_panel object, containing information about
                the used panels in the array
            render_dir (str): path to rendered roof images
            azimuth (float): azimuth orientation of the array
            inclination (float): inclination of the array
            resolution_I (int): Current (A) resolution for the simulation:
                Defaults to 110.
            start_year (int): Initial year for the simulation.
                Defaults to 2007.
            end_year (int): Last year for the simulation. Defaults to 2008.

        """
        self.shading = shading
        self.building = building
        self.pv_panel = pv_panel
        self.shading.building = building
        self.azimuth = azimuth
        self.inclination = inclination

        self.resolution_I = 110
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.grid = None
        self.start_year = start_year
        self.end_year = end_year
        self.timestep = building.timestep
        self.output_dir = None

    def set_offsets(self, offset_x: float = 0, offset_y: float = 0):
        """Set offsets of the solar array grid root.

        Sets the the offsets of the solar array with respect to the Shading
        images. With respect to the size of the used panel, the scale of the
        images and the offsets this function also creates an empty grid for
        possible panel locations

        Args:
            offset_x (float): Offset of the origin of the array along the
                x-Axis from the top left corner of the image. Defaults to 0.
            offset_y (float): Offset of the origin of the array along the
                y-Axis from the top left corner of the image. Defaults to 0.
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.grid = -1*np.ones(self.grid_shape).T

    def preprocess(self):
        """Preprocess the solar_array using provided information."""
        self.gen_panel_slices()
        self.get_resize_shading_fun()
        self.Pvgis = Pvgis(loc=self.building.location, azimuth=self.azimuth,
                           inclination=self.inclination,
                           startyear=self.start_year, endyear=self.end_year)

    def show_array(self,):
        """Preview Array configuration on roof using matplotlib."""
        shading = self.shading
        pv_panel = self.pv_panel
        offset_x, offset_y = self.offset_x, self.offset_y
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(shading.avg, extent=(
            0, shading.dim_x, shading.dim_y, 0), alpha=.7)
        if self.grid is None:
            grid_shape = self.grid_shape
            grid = -1*np.ones(grid_shape).T
            coords = np.argwhere(grid > -2)
        else:
            grid = self.grid
        coords = np.argwhere(grid > -2)
        ax.plot(self.offset_x, self.offset_y, 'x', markersize=20, color='red')
        for x, y in coords:
            string = int(grid[x, y])
            alpha = .5+.5*int(string > -1)
            rect = Rectangle((pv_panel.dim_x*x+offset_x,
                              pv_panel.dim_y*y+offset_y),
                             pv_panel.dim_x, pv_panel.dim_y,
                             fill=None, alpha=alpha)
            bound = rect.get_extents()
            center = np.average(bound, axis=0)
            ax.add_patch(rect)
            ax.text(*center, '({},{})\nStr: {}'.format(x, y, string),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12, color='black', alpha=alpha)
        return fig, ax

    @property
    def grid_shape(self):
        """Return a maximum possible grid shape.

        Depends on size and scale of the image and panel.
        """
        return (int((self.shading.dim_y-self.offset_y)//self.pv_panel.dim_y),
                int((self.shading.dim_x-self.offset_x)//self.pv_panel.dim_x))

    @property
    def cell_grid(self):
        """Return the actual cell grid shape depending on defined strings."""
        pv_p = self.pv_panel
        coords = np.argwhere(self.grid > 0)
        x0, y0 = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)+1
        return(self.grid[x0:x_max, y0:y_max].repeat(pv_p.cells_x, axis=0
                                                    ).repeat(
                                                        pv_p.cells_y, axis=1))

    def get_resize_shading_fun(self):
        """Set the image resize function."""
        assert not(self.grid is None), 'Grid needs to be defined first'
        cell_grid_shape = self.cell_grid.shape
        ind_y_start = int(self.offset_y*self.shading.scale)
        ind_x_start = int(self.offset_x*self.shading.scale)
        ind_y_delta = int(
            self.cell_grid.shape[1]*self.pv_panel.dim_y *
            self.shading.scale/self.pv_panel.cells_y)
        ind_x_delta = int(
            self.cell_grid.shape[0]*self.pv_panel.dim_x *
            self.shading.scale/self.pv_panel.cells_x)

        def resize(img):
            return zoom(img[ind_y_start:ind_y_start+ind_y_delta,
                            ind_x_start:ind_x_start+ind_x_delta, ],
                        zoom=[cell_grid_shape[1]/ind_y_delta,
                              cell_grid_shape[0]/ind_x_delta],
                        order=0)
        self.resize = resize

    def gen_panel_slices(self):
        """Set list of slices for each cell on a substring."""
        module_indices = []
        panels = np.argwhere(self.grid > 0)
        pv_panel = self.pv_panel
        for pos_x, pos_y in panels:
            x_min = pos_x*pv_panel.cells_x
            x_max = x_min
            substrings = []
            for bypass_x in pv_panel.bypass_config[0]:
                x_max += bypass_x
                y_min = pos_y*pv_panel.cells_y
                y_max = y_min
                if isinstance(pv_panel.bypass_config[1], int):
                    for bypass_y in [pv_panel.bypass_config[1]]:
                        y_max += bypass_y
                        substrings.append(
                            [slice(x_min, x_max), slice(y_min, y_max)])
                        y_min = y_max
                else:
                    for bypass_y in pv_panel.bypass_config[1]:
                        y_max += bypass_y
                        substrings.append(
                            [slice(x_min, x_max), slice(y_min, y_max)])
                        y_min = y_max
                x_min = x_max
            module_indices.append(substrings)
        self.module_indices = module_indices

    def show_panel_slices(self):
        """Short summary.

        Returns:
            type: .

        """
        data = np.zeros(self.cell_grid.T.shape)
        for ind, mod in enumerate(self.module_indices[:]):
            for ind2, substring in enumerate(mod):
                data[substring[1], substring[0]] += (ind2+1+ind*.3)
        plt.imshow(data)
        return data

    def run(self, Pvgis, start_date=None, end_date=None, n_threads=8,
            batch_size=24*6, ):
        """Short summary.

        Args:
            Pvgis (Pvgis): Pvgis data for the simulation.
            start_date (dt.datetime): Simulation startdate. Defaults to None.
            end_date (dt.datetime): . Simulation enddate to None.
            n_threads (int): . Defaults to 8.
            batch_size (int): . Defaults to 24*6.

        Returns:
            type: .

        """
        if start_date is None:
            start_date = dt.datetime(self.start_year, 1, 2)
        if end_date is None:
            end_date = dt.datetime(self.end_year, 1, 1)

        timestep = self.timestep
        steps = int((end_date.timestamp()-start_date.timestamp()) /
                    timestep/batch_size)
        batches = []
        for time in range(steps):
            curr_time = start_date + \
                dt.timedelta(seconds=time*timestep*batch_size)
            end_time = start_date + \
                dt.timedelta(seconds=(time+1)*timestep*batch_size)
            batches.append((curr_time, end_time))
        self.results = self.process(batches[0])
        # with Pool(n_threads) as p:
        #    self.batch_data = p.map(self.process, self.batches)

    def process(self, args):
        """Process function for use in multiprocessing.Pool.

        Args:
            args (List[dt.datetime, dt.datetime]): startdate and enddate of the
                simulation.

        Returns:
            type: .

        """
        timestep = self.timestep
        (start_date, end_date) = args
        steps = int(end_date.timestamp()-start_date.timestamp())//timestep
        I_conf = np.linspace(self.pv_panel.I_min,
                             self.pv_panel.I_max, self.resolution_I)
        n_modules = len(self.module_indices)
        V_I_data = np.zeros((steps, n_modules, self.resolution_I))
        for time in range(int(end_date.timestamp() -
                              start_date.timestamp())//timestep):
            curr_time = start_date+dt.timedelta(seconds=time*timestep)
            environment = self.Pvgis[curr_time]
            L_mult = environment['Bi']
            L_add_di = environment['Di']+environment['Ri']+environment['As']
            T_am = environment['Tamb']
            # Define Iph/G func more accurately (Right now it's just
            # a factor of 1e-3)
            W_10 = environment['W10']
            if L_add_di == 0:
                continue
            roof = self.resize(self.shading[curr_time])
            for ind, module in enumerate(self.module_indices):
                L_ = [np.reshape(roof[substring[0],
                                      substring[1]], -1
                                 )*L_mult*1e-3 +
                      L_add_di*1e-3 for substring in module]
                V_I_data[time, ind] = self.pv_panel.v_from_i(
                    (I_conf, L_, T_am, W_10, -.5))
            np.save("/home/clemens/data/VI2/" +
                    start_date.strftime('%Y-%m-%d'), V_I_data)
        return [start_date, end_date,  V_I_data]
