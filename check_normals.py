import os.path

import numpy as np
import matplotlib.pyplot as plt
from pyoctree import pyoctree as ot
import trimesh
import glob
import cv2
from tqdm import tqdm


def write_obj_file(path, vertices, vertex_colors, faces):
    with open(path, 'w') as f:
        for vertex, color in zip(vertices, vertex_colors):
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")

        for face in faces:
            # OBJ format uses 1-based indices, so we add 1 to each index
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
def load_K_Rt_from_P(P):

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


super_main_folder = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/"
all_f = glob.glob(super_main_folder+"*")
"""
for main_folder in all_f :
    main_folder = main_folder + "/"
    print(main_folder)
    methods = glob.glob(main_folder+"result_*")
    inputs = glob.glob(main_folder+"input*")
    all_folders = methods + inputs
    folder_GT = main_folder+"GT"

    # NORMALS GT
    path_normals_gt = glob.glob(folder_GT+"/normal/*.png")
    normals_gt = [(plt.imread(ngtp)*2.0-1.0)[:,:,:3] for ngtp in path_normals_gt]

    # MASK
    path_masks = glob.glob(folder_GT+"/mask/*.png")
    masks = [plt.imread(ngtp) for ngtp in path_masks]

    camera_data = np.load(main_folder+"/cameras.npz")
    nb_views = len(camera_data.files)//2
    lk = []
    lr = []
    lt = []
    for k in range(nb_views):
        P_k = camera_data["world_mat_{}".format(k)]
        K,RT = load_K_Rt_from_P(P_k[:3,:])
        lr.append(RT[:3,:3].T)
        lt.append(-RT[:3,:3].T @ RT[:3,[3]])
        lk.append(K[:3,:3])

    all_normals = []
    all_geodesic = []
    for folder in all_folders :
        print("    > {}".format(folder))
        path_normals_folder = glob.glob(folder+"/normal/*.png")
        normals_folder = [(plt.imread(ngtp)*2.0-1.0)[:,:,:3] for ngtp in path_normals_folder]
        all_normals.append(normals_folder)

        path_mask_folder = glob.glob(folder + "/mask/*.png")
        mask_folder = [plt.imread(ngtp) for ngtp in path_mask_folder]
        if len(mask_folder[0].shape)>2:
            for n in range(len(mask_folder)):
                mask_folder[n] = mask_folder[n][:,:,0]

        geodesic_output_folder = folder+"/geodesic_distance/"
        if not os.path.exists(geodesic_output_folder):
            os.mkdir(geodesic_output_folder)


        geodesic_folder = []
        for k in range(len(normals_folder)):
            mask = masks[k][:, :, 0]
            mask_fold = mask_folder[k]
            mask = mask * mask_fold
            ind = np.where(mask < 0.5)
            n1 = normals_folder[k] / np.linalg.norm(normals_folder[k],axis=2,keepdims=True)
            n2 = normals_gt[k] / np.linalg.norm(normals_gt[k],axis=2,keepdims=True)
            dot_n = (n1*n2).sum(axis=2).clip(-1.0,1.0)
            geodesic_distance = np.arccos(dot_n)
            geodesic_distance /= np.pi
            geodesic_distance[ind] = 0.0
            geodesic_distance[0,0] = 1.0
            plt.imsave(geodesic_output_folder+(3-len(str(k)))*"0"+str(k)+".jpg",geodesic_distance)
            geodesic_folder.append(geodesic_distance)
        all_geodesic.append(geodesic_folder)
"""
"""
for main_folder in all_f[2:] :

    folder_GT = main_folder + "/GT/"

    # MESH GT OCTREE GENERATION

    pathObjGT2 = glob.glob(folder_GT + "mesh.ply")[0]
    mesh2 = trimesh.load(pathObjGT2)
    vertices2 = np.array(mesh2.vertices)

    pathObjGT = glob.glob(folder_GT+"mesh2.ply")[0]
    camera_data = np.load(main_folder + "/cameras.npz")
    nb_views = len(camera_data.files) // 2
    print("before load")
    mesh = trimesh.load(pathObjGT)
    #rotation = trimesh.transformations.rotation_matrix(
    #    np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces).astype(np.int32)
    print("before octree")
    octree = ot.PyOctree(vertices, faces)
    count_vertices = np.zeros(vertices.shape[0])
    for view in range(nb_views):

        mask = plt.imread(folder_GT+"/mask/"+"0"*(3-len(str(view)))+str(view)+".png")[:,:,0]

        P_k = camera_data["world_mat_{}".format(view)]
        K, RT = load_K_Rt_from_P(P_k[:3, :])
        R = RT[:3,:3].T
        T = -RT[:3,:3].T @ RT[:3,[3]]
        K = K[:3,:3]

        vert_cam = R@vertices.T + T
        vert_cam /= vert_cam[2,:]
        vert_pix = K @ vert_cam
        vert_pix = np.round(vert_pix).astype(int)[:2,:]

        #ind = mask[vert_pix[1,:],vert_pix[0,:]].astype(bool)
        #vert_in_mask = vertices2.T[:,ind]


        origins = vertices.T
        cam_center = -R.T @ T

        for nvert in tqdm(range(origins.shape[1])):
            a = origins[:,nvert]
            b = cam_center.reshape(3)
            ray = np.array([a, b]).astype(np.float32)
            result = octree.rayIntersection(ray)
            intersection = False
            if len(result) != 0:
                t = [result[v].s for v in range(len(result))]
                if np.max(t) > 0.01 :
                    intersection = True
            if not intersection :
                count_vertices[nvert] += 1

    count_vertices = count_vertices.reshape(-1,1)
    count_vertices = np.concatenate((count_vertices,count_vertices,count_vertices),axis=1)
    count_vertices = count_vertices/nb_views
    write_obj_file(folder_GT+"/mesh_visibility.obj",vertices,count_vertices,faces)
        #mesh.export(folder_GT+"/mesh_visibility.ply", file_type='ply')
    a=0
"""


for main_folder in all_f :

    folder_GT = main_folder + "/GT/"

    # MESH GT OCTREE GENERATION

    pathObjGT2 = glob.glob(folder_GT + "mesh.ply")[0]
    mesh2 = trimesh.load(pathObjGT2)
    vertices2 = np.array(mesh2.vertices)

    pathObjGT = glob.glob(folder_GT+"mesh2.ply")[0]
    camera_data = np.load(main_folder + "/cameras.npz")
    nb_views = len(camera_data.files) // 2
    print("before load")
    mesh = trimesh.load(pathObjGT)
    #rotation = trimesh.transformations.rotation_matrix(
    #    np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces).astype(np.int32)
    print("before octree")
    octree = ot.PyOctree(vertices, faces)
    count_vertices = np.zeros(vertices.shape[0])
    dist_vertices = [[] for k in range(len(count_vertices))]
    simil_norm_vertices = np.zeros(vertices.shape[0])
    # NORMALS GT
    path_normals_gt = glob.glob(folder_GT + "/normal/*.png")
    normals_gt = [(plt.imread(ngtp) * 2.0 - 1.0)[:, :, :3] for ngtp in path_normals_gt]

    path_normals_folder = glob.glob(main_folder + "/input_SDM/normal/*.png")
    normals_folder = [(plt.imread(ngtp) * 2.0 - 1.0)[:,:,:3] for ngtp in path_normals_folder]

    for view in range(nb_views):

        norm_gt = normals_gt[view]
        norm_sdm = normals_folder[view]
        #print(main_folder+"/input_SDM/mask/"+"0"*(3-len(str(view)))+str(view)+".png")
        mask = plt.imread(main_folder+"/input_SDM/mask/"+"0"*(3-len(str(view)))+str(view)+".png")

        P_k = camera_data["world_mat_{}".format(view)]
        K, RT = load_K_Rt_from_P(P_k[:3, :])
        R = RT[:3,:3].T
        T = -RT[:3,:3].T @ RT[:3,[3]]
        K = K[:3,:3]

        vert_cam = R@vertices.T + T
        vert_cam /= vert_cam[2,:]
        vert_pix = K @ vert_cam
        vert_pix = np.round(vert_pix).astype(int)[:2,:]

        #ind = mask[vert_pix[1,:],vert_pix[0,:]].astype(bool)
        #vert_in_mask = vertices2.T[:,ind]


        origins = vertices.T
        cam_center = -R.T @ T

        for nvert in tqdm(range(origins.shape[1])):
            a = origins[:,nvert]
            b = cam_center.reshape(3)
            ray = np.array([a, b]).astype(np.float32)
            result = octree.rayIntersection(ray)
            intersection = False
            if len(result) != 0:
                t = [result[v].s for v in range(len(result))]
                if np.max(t) > 0.01 :
                    intersection = True
            if not intersection :
                count_vertices[nvert] += 1
                vertex_pixel = vert_pix[:,nvert]
                normal_pixel = norm_sdm[vertex_pixel[1],vertex_pixel[0]]
                dist_vertices[nvert].append(normal_pixel)


    for k in range(len(dist_vertices)):

        all = dist_vertices[k]
        all_n = []
        for kk in range(len(all)):
            if np.linalg.norm(all[kk])>0.90 :
                all_n.append(all[kk])
        if len(all_n)>1 :
            dist = 0
            for i in range(len(all_n)):
                n1 = all_n[i] / np.linalg.norm(all_n[i])
                for j in range(i+1,len(all_n)):
                    n2 = all_n[j] / np.linalg.norm(all_n[j])
                    dot_n = (n1 * n2).sum().clip(-1.0, 1.0)
                    geodesic_distance = np.arccos(dot_n)/np.pi
                    dist += geodesic_distance
            dist /= (len(all_n)*(len(all_n)-1)/2)
            simil_norm_vertices[k] = dist


    simil_norm_vertices = simil_norm_vertices.reshape(-1,1)
    simil_norm_vertices = np.concatenate((simil_norm_vertices,simil_norm_vertices,simil_norm_vertices),axis=1)
    write_obj_file(main_folder+"/input_SDM/mesh_normal_similarity.obj",vertices,simil_norm_vertices,faces)
    a=0

