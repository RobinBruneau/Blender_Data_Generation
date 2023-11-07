import subprocess
import os
import argparse
from camera_npz_to_npy import *
from world_to_camera_normals import *

path_code_python = "blender_event_data_generation.py"

if __name__ == '__main__':

    main_folder = "D:/PhD/Dropbox/CVPR_2024_2/results/"
    sub_folders= glob.glob(main_folder+"*")

    for folder in sub_folders :

        cam_file = folder + "/cameras.npz"

        methods = glob.glob(folder+"/result_*")
        for meth in methods :

            mesh_file = glob.glob(meth+"/*.ply")
            if len(mesh_file) == 0 :
                mesh_file = glob.glob(meth+"/*.obj")[0]
            else :
                mesh_file = mesh_file[0]
            output_folder = meth+"/"

            #if not os.path.exists(output_folder+"cameras_unpacked.npz"):

            # 1 : unpacked cameras.npz
            cameras_npz_unpacked(output_folder,cam_file)

            # 2 : Generate normals per view
            os.system("blender --enable-event-simulate --python {} -- --folder {} --mesh {} ".format(path_code_python,output_folder,mesh_file))


