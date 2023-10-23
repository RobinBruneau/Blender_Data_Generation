import subprocess
import os
import argparse
from camera_npz_to_npy import *
from world_to_camera_normals import *

path_code_python = "blender_event_data_generation.py"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default="")
    parser.add_argument('--mesh', type=str, default="")
    parser.add_argument('--output_folder', type=str, default="")
    args = parser.parse_args()
    cam_file = args.cam
    mesh_file = args.mesh
    output_folder = args.output_folder

    if not os.path.exists(cam_file):
        raise("cameras.npz file path doesn't exist !")
    if not os.path.exists(mesh_file):
        raise("mesh file path doesn't exist !")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 1 : unpacked cameras.npz
    cameras_npz_unpacked(output_folder,cam_file)

    # 2 : Generate normals per view
    #os.system("blender")
    #os.system("blender -b --python {}".format(path_code_python))
    os.system("blender --enable-event-simulate --python {} -- --folder {} --mesh {} ".format(path_code_python,output_folder,mesh_file))
    #subprocess.check_call("C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe --enable-event-simulate --python {} ".format(path_code_python))
    #subprocess.check_call("C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe -b --python {} -- --mesh ici".format(path_code_python))

    # subprocess.check_call("./blender --enable-event-simulate --python {} ".format(path_code_python))

    # 3 : set normal maps in cameras views
    world_to_cam_normals(output_folder)


