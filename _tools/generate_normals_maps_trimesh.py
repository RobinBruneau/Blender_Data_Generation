"""
Do simple mesh- ray queries. Base functionality only
requires numpy, but if you install `pyembree` you get the
same API with a roughly 50x speedup.
"""
import copy

import matplotlib.pyplot as plt
import trimesh
import numpy as np
import math
import cv2
from tqdm import tqdm

if __name__ == '__main__':


    folder = "D:\PhD\Projects\Blender_Data_Generation\OUTPUT\Data_xyz_normal/"

    mask = cv2.imread(folder+"mask.png").sum(axis=2)
    im_normal = cv2.imread(folder+"mask.png")
    pixels =  np.asarray(np.where(mask > 200))
    pixels = pixels[-1::-1, :]
    line = np.ones((1,pixels.shape[1]))
    pixels = np.concatenate((pixels,line),axis=0)


    R = np.load(folder+"R.npy").reshape(3,3)
    T = np.load(folder+"T.npy")
    K = np.load(folder + "K.npy")
    cam_world = -R.transpose() @ T

    pixels_cam = np.linalg.inv(K) @ pixels
    pixels_world = R.T @ (pixels_cam - T)

    # test on a sphere mesh
    pathObj = "D:/INPUT/Graphosoma/Graphosoma.obj"
    mesh = trimesh.load(pathObj)

    rotation = trimesh.transformations.rotation_matrix(
        np.deg2rad(90), [1, 0, 0], point=[0,0,0])
    mesh.apply_transform(rotation)

    print(np.min(mesh.vertices[:,0]),np.max(mesh.vertices[:,0]))
    print(np.min(mesh.vertices[:, 1]), np.max(mesh.vertices[:, 1]))
    print(np.min(mesh.vertices[:, 2]), np.max(mesh.vertices[:, 2]))
    # create some rays


    nb = pixels_world.shape[1]
    indd = [i for i in range(pixels_world.shape[1])]
    ind_ = np.array_split(indd,pixels_world.shape[1]//10000 + 1)

    for ind in tqdm(ind_) :
        pixels_im_ind = pixels[:2, ind].T
        pixels_final = pixels_im_ind.astype(int)
        im_normal[pixels_final[:, 1], pixels_final[:, 0], 1] = 100
    plt.imsave(folder + "normal_check.png", im_normal)
    for ind in tqdm(ind_):
        pixels_im_ind = pixels[:2,ind].T
        pixels_ind = pixels_world[:,ind].T
        ray_origins = np.repeat(cam_world,len(ind),1).T
        #ray_origins = ray_origins / np.linalg.norm(ray_origins,axis=1).reshape(-1,1)
        ray_directions = pixels_ind - ray_origins


        """
        Signature: mesh.ray.intersects_location(ray_origins,
                                                ray_directions,
                                                multiple_hits=True)
        Docstring:
    
        Return the location of where a ray hits a surface.
    
        Parameters
        ----------
        ray_origins:    (n,3) float, origins of rays
        ray_directions: (n,3) float, direction (vector) of rays
    
    
        Returns
        ---------
        locations: (n) sequence of (m,3) intersection points
        index_ray: (n,) int, list of ray index
        index_tri: (n,) int, list of triangle (face) indexes
        """

        # run the mesh- ray test
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False)


        # stack rays into line segments for visualization as Path3D
        ray_visualize = trimesh.load_path(np.hstack((
            ray_origins,
            ray_origins + ray_directions)).reshape(-1, 2, 3))

        faces_hit = mesh.faces[index_tri,:]
        normal = 255*(mesh.face_normals[index_tri,:]+1)/2.0
        pixels_final = pixels_im_ind[index_ray,:].astype(int)
        im_normal[pixels_final[:,1],pixels_final[:,0],:] = normal
    plt.imsave(folder+"normal_cmp.png",im_normal)

    '''
    v1 = mesh.vertices[faces_hit[:,0]]
    v2 = mesh.vertices[faces_hit[:,1]]
    v3 = mesh.vertices[faces_hit[:,2]]

    v12_visualize = trimesh.load_path(np.hstack((
        v1,v2)).reshape(-1, 2, 3))

    v23_visualize = trimesh.load_path(np.hstack((
        v2,v3)).reshape(-1, 2, 3))

    v31_visualize = trimesh.load_path(np.hstack((
        v3,v1)).reshape(-1, 2, 3))

    # make mesh transparent- ish
    mesh.visual.face_colors = [100, 100, 100, 100]

    # create a visualization scene with rays, hits, and mesh
    scene = trimesh.Scene([
        ray_visualize,
        v12_visualize,
        v23_visualize,
        v31_visualize,
        trimesh.points.PointCloud(locations)])

    # display the scene
    scene.show()
    '''

