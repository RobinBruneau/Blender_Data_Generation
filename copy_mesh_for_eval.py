import os
import shutil
import glob

to_copy = "D:/PhD/Dropbox/CVPR_2024/RESULTS_MESHES/"
to_paste = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/"

cases = ["bearPNG","buddhaPNG","cowPNG",'pot2PNG',"readingPNG"]

for case in cases :

    main_folder_copy = to_copy + case +"/"
    main_folder_paste = to_paste + case + "/"

    folders_to_copy = glob.glob(main_folder_copy+"proposed*")

    for folder in folders_to_copy :
        name = folder.split("roposed_")[1]
        shutil.copytree(folder,main_folder_paste+"result_Ours_({})".format(name))