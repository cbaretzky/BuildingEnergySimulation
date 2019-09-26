import os
from pathlib import Path
import numpy as np
import tempfile
import subprocess
from sys import platform
from getpass import getuser
from . import solar


class Render_manager():
    """
    Module to manage the data on disk for:
     - shading calculations
     - calculated power curves

     az_delta = angle between x axis in blender file and north
    """

    def __init__(self,building, blend_file, az_delta = 0):
        temp_dir = Path(tempfile.gettempdir())
        self.location = building.location
        self.horizon = building.horizon
        self.suns = solar.SunData(self.location, self.horizon)
        self.blend_file = Path(blend_file)
        self.render_dir = self.blend_file.parent
        self.path_to_blender = self.get_path_to_blender()
        return_code = subprocess.call(self.path_to_blender+" -b", shell=True)
        self.render_angles = self.get_render_angles(self.suns)
        self.az_delta = az_delta
        if not return_code is 0:
            raise FileNotFoundError("Could not run blender through '{}''".format(self.path_to_blender))
    def run(self):
        self.render(self.get_render_angles, self.blend_file, self.path_to_blender)
    @staticmethod
    def get_path_to_blender():
        """
        Get the blender executable on Linux, macOS and Windows
        """
        if platform.startswith('freebsd'):
            return "blender"
        elif platform.startswith('linux'):
            return "blender"
        elif platform.startswith('darwin'):
            return Path("/Applications/blender/Contents/MacOS/blender")
        elif platform.startswith('win32'):
            return Path(r'C:\Program Files\Blender Foundation\Blender\blender.exe')
    @staticmethod
    def get_render_angles(suns):
        """
        return set of integer tuples from sun positions
        """
        return set((angle[0], angle[1]) for angle in np.asarray(suns.data[['azimuth','elevation', "is_up"]]).astype(np.int) if angle[2])

    @staticmethod
    def render(render_angles, blendfile, path_to_blender, render_dir="render/", az_delta=0,):
        """
        Managing illumination render
        blendfile = path to blender file
        render_dir = path to output renders
        rendered_dir = path of rendered images (defaults to render_dir)
        """

        #setting up directories
        blendfile = Path(blendfile)
        if isinstance(render_dir, str):
            render_dir = Path(render_dir)
        render_dir = Path(render_dir)
        if not blendfile.exists():
            raise FileNotFoundError('blendfile at '+str(blendfile.absolute())+' does not exist')
        if not os.path.isdir(render_dir):
            try:
                os.mkdir(render_dir)
            except Exception as e:
                print(e)
                raise IOError("Couldn't create {}".format(render_dir.absolute()))

        angles = render_angles
        config_file_path = Path.joinpath(render_dir,Path('config_file'))
        """
        Writing config and commands for blender
        """
        angles = render_angles
        config_file = []
        imports_setup = """
import bpy
import os
import time
from mathutils import *
from math import *
from pathlib import Path
file_name_blend = Path(bpy.data.filepath)

start_time = time.time()
sun_ob = bpy.data.objects['Sun']
cam = bpy.context.scene.camera
rot = [str(int(r*180/3.141592645)) for r in cam.rotation_euler]
fname="_".join([file_name_blend.name.split('.')[0],*rot])
"""
        config_file.append(imports_setup)

        t_dir = "target_dir = '{}'".format(
        str(Path.joinpath(render_dir, Path(blendfile.stem+hex(hash(tuple(angles))))).absolute())
        )

        config_file.append(t_dir)

        t_angles=  "angles = "+str(list(angles))
        angle_offset="az_delta = {}".format(az_delta)
        config_file.append(t_angles)
        config_file.append("\n")
        config_file.append(angle_offset)
        config_file.append("\n")

        render_exe = r"""
target_dir = target_dir
bpy.context.scene.render.filepath = target_dir
print(Path.joinpath(Path(target_dir), Path(fname)))
for angle in angles:
    az = angle[0]
    al = angle[1]
    sun_ob.rotation_euler = Euler((0, (-90+al)*pi/180,(-az_delta-az)*pi/180 ))

    imgname = str(az)+"_"+str(al)+".png"
    img_path = str(Path.joinpath(Path(target_dir),Path(imgname)).absolute())
    bpy.data.scenes['Scene'].render.filepath = img_path
    bpy.ops.render.render( write_still=True )
    errc  = 0
    while errc<30:
        try:
            os.path.getsize(img_path)
            print("errc = {}".format(errc))
            break
        except:
            time.sleep(.01)
        errc +=1
        if errc > 100:
            print('Could not read file: {}'.format(img_path))
            break
print("Rendered {} images in {} seconds".format(len(angles), time.time()-start_time))
        """
        config_file.append(render_exe)

        with open(config_file_path,'w') as f:
            f.write("\n".join(config_file))
        command = path_to_blender+' '+str(blendfile.absolute())+' -b -P '+str(Path(config_file_path).absolute())
        print(command)
        return_code = subprocess.call(command, shell=True)
