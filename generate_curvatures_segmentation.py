import glob
import os.path

import trimesh
import matplotlib.pyplot as plt
import numpy as np

def decodeur_colormap(color,val_min,val_max):
    r,g,b = color
    if r > 0.99 :
        r = 1.0
    if g > 0.99 :
        g = 1.0
    if b > 0.99 :
        b = 1.0
    if g == 0.0 and b >=0.5 :
        t = (b-0.5)/4.0
    elif b==1.0 :
        t = g/4.0 + 0.125
    elif g==1.0 :
        t = r/4.0 + 0.375
    elif r==1.0 :
        t = (1-g)/4.0 + 0.625
    elif g==0 and b==0 :
        t = (1-r)/4.0 + 0.875
    else :
        print(color)
        t=0
    return val_min + (val_max - val_min) * t


main = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/"
cases = ["bearPNG","buddhaPNG","cowPNG",'pot2PNG',"readingPNG"]


for case in cases :

    curv_min_path = glob.glob(main + case + "/GT_curvature/*cmin.obj")[0]
    curv_max_path = glob.glob(main + case + "/GT_curvature/*cmax.obj")[0]


    mesh_cmin = trimesh.load_mesh(curv_min_path)
    mesh_cmax = trimesh.load_mesh(curv_max_path)

    vert_color_cmin = np.array(mesh_cmin.visual.vertex_colors)[:,:3]
    vert_color_cmin = vert_color_cmin/255.0
    vert_cmin = mesh_cmin.vertices

    vert_color_cmax = np.array(mesh_cmax.visual.vertex_colors)[:, :3]
    vert_color_cmax = vert_color_cmax / 255.0
    vert_cmax = mesh_cmax.vertices


    color_decoded_cmin = []
    for k in range(vert_color_cmin.shape[0]):
        v = decodeur_colormap(vert_color_cmin[k,:],0,1)
        color_decoded_cmin.append(v)
    color_decoded_cmin = np.array(color_decoded_cmin)

    color_decoded_cmax = []
    for k in range(vert_color_cmax.shape[0]):
        v = decodeur_colormap(vert_color_cmax[k, :], 0, 1)
        color_decoded_cmax.append(v)
    color_decoded_cmax = np.array(color_decoded_cmax)


    ind_convexity = color_decoded_cmin<=0.125

    ind_concavity = color_decoded_cmax>=0.875


    mesh_cmin.visual.vertex_colors[:,:3] = (0,255,0)
    mesh_cmin.visual.vertex_colors[ind_convexity, :3] = (0, 0, 255)
    mesh_cmin.visual.vertex_colors[ind_concavity,:3] = (255,0,0)
    mesh_cmin.export(main + case + "/GT_curvature/curvature_seg_2.obj")

