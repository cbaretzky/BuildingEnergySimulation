import os
from pathlib import Path
import subprocess
from sys import platform
from getpass import getuser

class file_man():
    """
    Module to manage the data on disk for:
     - shading calculations
     - calculated power curves
    """

    def __init__(self, project_dir):
        raise NotImplemented("This is still WIP")
        self.project_dir = Path(project_dir)
        self.blend_file = None
        self.render_dir = "render/"
        self.rendered_dir = self.render_dir

def path_to_blender():
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

def get_render_angles(suns):
    """
    return set of tuples from sun positions
    """
    angles = []
    bpy_angles = []
    for stamp, sun in suns.items():
        if sun.up:
            angles.append(tuple(int(x) for x in sun.az_al))
    return angles

def blend_angle(angle =(0,13), az_delta = -44):
    """
    Calculate
    """
    az = angle[0]
    al = angle[1]
    #return (0, (-90+al)*pi/180,-(az_delta-az) *pi/180)
    return (0, -90+al,(-az_delta-az) )


def render(suns, blendfile = "untitled.blend", render_dir="render/", rendered_dir = None):
    """
    Managing illumination render
    blendfile = path to blender file
    render_dir = path to output renders
    rendered_dir = path of rendered images (defaults to render_dir)
    """

    #setting up directories
    blendfile = Path(blendfile)
    render_dir = Path(render_dir)
    if not rendered_dir:
        rendered_dir  = render_dir
    if not blendfile.exists():
        raise FileNotFoundError('blendfile at '+str(blendfile.absolute())+' does not exist')
    if not os.path.isdir(render_dir):
        try:
            os.mkdir(render_dir)
        except Exception as e:
            print(e)
            return(0)

    angles = set(get_render_angles(suns))
    files_ = set(os.listdir())
    ang_from_file = set([ (int(file[:-4].split('_')[0]), int(file[:-4].split('_')[1])) for file in files_ ])
    angles = set(get_render_angles(suns))- ang_from_file



    """
    Writing config and commands for blender
    """
    files_ = set(os.listdir(rendered_dir))
    for file in files:
        ang_from_file = []
        try:
            ang_from_file.append((int(file[:-4].split('_')[0]), int(file[:-4].split('_')[1])))
        except:
            print("Failed on"+file)
        ang_from_file = set(ang_from_file)
    ang_from_file = set([ (int(file[:-4].split('_')[0]), int(file[:-4].split('_')[1])) for file in files_ ])
    angles = set(get_render_angles(suns))- ang_from_file
    config_file = []
    imports = """import bpy
    import os
    import json
    import time
    from mathutils import *
    from math import *

    sun_ob = bpy.data.objects['Sun']
    """
    config_file.append(imports)
    t_dir = "target_dir = '{}'".format(str(target_dir.absolute()))
    config_file.append(t_dir)

    t_angles=  "angles = "+str(list(angles))
    config_file.append(t_angles)

    render_exe = """
    for angle in angles:
        az = angle[0]
        al = angle[1]
        az_delta = -44
        sun_ob.rotation_euler = Euler((0, (-90+al)*pi/180,(-az_delta-az)*pi/180 ))
        fname = os.path.join(target_dir,str(az)+"_"+str(al)+".png")
        bpy.data.scenes['Scene'].render.filepath = fname
        bpy.ops.render.render( write_still=True )
        errc  = 0
        while not(str(az)+"_"+str(al)+".png" in os.listdir(target_dir)):
            time.sleep(.001)
            errc +=1
            if errc > 10:
                break
        """
    config_file.append(render_exe)

    with open(config_file_path,'w') as f:
        f.write("\n".join(config_file))
    command = path_to_blender+' '+str(blendfile.absolute())+'-b -p '+';'.join(config_file)
    return_code = subprocess.call("", shell=True)
    os.system(path_to_blender()+' '+blend_file+" -b -P "+config_file_path)
