import subprocess
import os
import argparse
from camera_npz_to_npy import *
from world_to_camera_normals import *

path_code_python = "blender_event_data_generation_vertex_color_curv.py"

if __name__ == '__main__':

    main_folder = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/"
    sub_folders= glob.glob(main_folder+"*")

    for folder in sub_folders :

        cam_file = folder + "/cameras.npz"

        methods = glob.glob(folder+"/GT_curv*")
        for meth in methods :

            mesh_file = glob.glob(meth+"/*curvature_seg.obj")
            if len(mesh_file) == 0 :
                mesh_file = glob.glob(meth+"/*.obj")[0]
            else :
                mesh_file = mesh_file[0]
            output_folder = meth+"/"
            output = "curvature"

            # 1 : unpacked cameras.npz
            cameras_npz_unpacked(output_folder,cam_file)

            # 2 : Generate normals per view
            os.system("blender --enable-event-simulate --python {} -- --folder {} --mesh {} --output {}".format(path_code_python,output_folder,mesh_file,output))


