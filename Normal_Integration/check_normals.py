import concurrent.futures
import os.path
import shutil
from functools import partial
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


def mae() :
    super_main_folder = "D:/PhD/Dropbox/CVPR_2024_2/results/"
    all_f = glob.glob(super_main_folder + "*")

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

            geodesic_crop_output_folder = folder + "/geodesic_distance_crop/"
            if not os.path.exists(geodesic_crop_output_folder):
                os.mkdir(geodesic_crop_output_folder)

            MAE_folder = folder + "/MAE/"
            if not os.path.exists(MAE_folder):
                os.mkdir(MAE_folder)


            geodesic_folder = []
            for k in range(len(normals_folder)):
                mask = masks[k]
                mask_fold = mask_folder[k]
                mask2 = mask + mask_fold
                ind_remove = np.where(mask < 0.5)
                ind_keep = np.where(mask>0.5)
                ind_bad = np.where(mask2 == 1.0)
                n1 = normals_folder[k] / np.linalg.norm(normals_folder[k],axis=2,keepdims=True)
                n2 = normals_gt[k] / np.linalg.norm(normals_gt[k],axis=2,keepdims=True)
                dot_n = (n1*n2).sum(axis=2).clip(-1.0,1.0)
                geodesic_distance = np.arccos(dot_n)

                MAE_wo_missing = np.copy(geodesic_distance)
                MAE_wo_missing[ind_bad[0],ind_bad[1]] = 0.0
                MAE_wo_missing[ind_remove[0], ind_remove[1]] = 0.0
                nb_wo_missing = len(ind_keep[0])-len(ind_bad[0])
                MAE_wo_missing = np.sum(MAE_wo_missing)/nb_wo_missing

                geodesic_distance = geodesic_distance.clip(0.0,np.pi/4)
                geodesic_distance /= (np.pi/4)
                geodesic_distance[ind_bad[0],ind_bad[1]] = 1.0
                distances_color = plt.cm.jet(geodesic_distance)[:, :, :3]
                distances_color[ind_remove[0],ind_remove[1],:] = 1
                distances_color_crop = distances_color[np.min(ind_keep[0]):np.max(ind_keep[0]),np.min(ind_keep[1]):np.max(ind_keep[1]),:]
                plt.imsave(geodesic_output_folder+(3-len(str(k)))*"0"+str(k)+".jpg",distances_color)
                plt.imsave(geodesic_crop_output_folder + (3 - len(str(k))) * "0" + str(k) + ".jpg", distances_color_crop)
                np.save(MAE_folder + (3 - len(str(k))) * "0" + str(k) + ".npy",MAE_wo_missing)


                geodesic_folder.append(geodesic_distance)
            all_geodesic.append(geodesic_folder)


def mae_visibility() :
    super_main_folder = "D:/PhD/Dropbox/CVPR_2024_2/results/"
    all_f = glob.glob(super_main_folder + "*")

    for main_folder in all_f :
        main_folder = main_folder + "/"
        print(main_folder)
        methods = glob.glob(main_folder+"result_*")
        inputs = glob.glob(main_folder+"input*")
        all_folders = methods + inputs
        folder_GT = main_folder+"GT"
        folder_GT_vis = main_folder + "GT_visibility"
        folder_GT_curv = main_folder + "GT_curvature"

        # NORMALS GT
        path_normals_gt = glob.glob(folder_GT+"/normal/*.png")
        normals_gt = [(plt.imread(ngtp)*2.0-1.0)[:,:,:3] for ngtp in path_normals_gt]

        # VISIBILITY GT
        path_visibility_gt = glob.glob(folder_GT_vis + "/visibility/*.png")
        visibility_gt = [(plt.imread(ngtp) * 2.0 - 1.0)[:, :, :3] for ngtp in path_visibility_gt]

        # CURVATURE GT
        path_curvature_gt = glob.glob(folder_GT_curv + "/curvature/*.png")
        curvature_gt = [(plt.imread(ngtp) * 2.0 - 1.0)[:, :, :3] for ngtp in path_curvature_gt]

        # MASK GT
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


            MAE_folder = folder + "/MAE/"
            if not os.path.exists(MAE_folder):
                os.mkdir(MAE_folder)

            MAE_vis_folder = folder + "/MAE_vis/"
            if os.path.exists(MAE_vis_folder):
                shutil.rmtree(MAE_vis_folder)
            os.mkdir(MAE_vis_folder)

            MAE_curv_folder = folder + "/MAE_curv/"
            if os.path.exists(MAE_curv_folder):
                shutil.rmtree(MAE_curv_folder)
            os.mkdir(MAE_curv_folder)


            geodesic_folder = []
            for k in range(len(normals_folder)):
                mask = masks[k]
                mask_fold = mask_folder[k]
                mask2 = mask + mask_fold
                ind_keep = np.where(mask2 == 2.0)
                ind_remove = np.where(mask2 < 2.0)
                n1 = normals_folder[k] / np.linalg.norm(normals_folder[k],axis=2,keepdims=True)
                n2 = normals_gt[k] / np.linalg.norm(normals_gt[k],axis=2,keepdims=True)
                dot_n = (n1*n2).sum(axis=2).clip(-1.0,1.0)
                geodesic_distance = np.arccos(dot_n)

                MAE = np.copy(geodesic_distance)
                MAE[ind_remove[0],ind_remove[1]] = 0.0
                nb = len(ind_keep[0])
                #MAE = np.sum(MAE)/nb

                visibility = visibility_gt[k][ind_keep]
                curvature = curvature_gt[k][ind_keep]

                # bleu -> rouge
                #c1 = [0.0,0.5,0.5]
                c2 = np.array([0.0,0.5,1.0]).reshape(1,3)
                c3 = np.array([0.5,1.0,0.5]).reshape(1,3)
                c4 = np.array([1.0,0.5,0.0]).reshape(1,3)
                c5 = np.array([0.5,0.0,0.0]).reshape(1,3)

                dist_c2 = ((visibility - c2) ** 2).sum(axis=1,keepdims=True)
                dist_c3 = ((visibility - c3) ** 2).sum(axis=1,keepdims=True)
                dist_c4 = ((visibility - c4) ** 2).sum(axis=1,keepdims=True)
                dist_c5 = ((visibility - c5) ** 2).sum(axis=1,keepdims=True)

                all_dist = np.concatenate((dist_c2,dist_c3,dist_c4,dist_c5),axis=1)
                ind_color = np.argmin(all_dist,axis=1)
                a=0

                ind_c2 = np.where(ind_color == 0)[0]
                if len(ind_c2) == 0 :
                    MAE_c2 = -1
                else :
                    MAE_c2 = MAE[ind_keep[0][ind_c2],ind_keep[1][ind_c2]]
                    MAE_c2 = np.mean(MAE_c2)
                    np.save(MAE_vis_folder + (3 - len(str(k))) * "0" + str(k) + "_zone1.npy", MAE_c2)

                ind_c3 = np.where(ind_color == 1)[0]
                if len(ind_c3) == 0 :
                    MAE_c3 = -1
                else :
                    MAE_c3 = MAE[ind_keep[0][ind_c3], ind_keep[1][ind_c3]]
                    MAE_c3 = np.mean(MAE_c3)
                    np.save(MAE_vis_folder + (3 - len(str(k))) * "0" + str(k) + "_zone2.npy", MAE_c3)

                ind_c4 = np.where(ind_color == 2)[0]
                if len(ind_c4) == 0 :
                    MAE_c4 = -1
                else :
                    MAE_c4 = MAE[ind_keep[0][ind_c4], ind_keep[1][ind_c4]]
                    MAE_c4 = np.mean(MAE_c4)
                    np.save(MAE_vis_folder + (3 - len(str(k))) * "0" + str(k) + "_zone3.npy", MAE_c4)

                ind_c5 = np.where(ind_color == 3)[0]
                if len(ind_c5) == 0 :
                    MAE_c5 = -1
                else :
                    MAE_c5 = MAE[ind_keep[0][ind_c5], ind_keep[1][ind_c5]]
                    MAE_c5 = np.mean(MAE_c5)
                    np.save(MAE_vis_folder + (3 - len(str(k))) * "0" + str(k) + "_zone4.npy", MAE_c5)

                c2 = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
                c3 = np.array([0.0, 1.0, 0.0]).reshape(1, 3)
                c4 = np.array([0.0, 0.0, 1.0]).reshape(1, 3)

                dist_c2 = ((visibility - c2) ** 2).sum(axis=1, keepdims=True)
                dist_c3 = ((visibility - c3) ** 2).sum(axis=1, keepdims=True)
                dist_c4 = ((visibility - c4) ** 2).sum(axis=1, keepdims=True)

                all_dist = np.concatenate((dist_c2, dist_c3, dist_c4), axis=1)
                ind_color = np.argmin(all_dist, axis=1)

                ind_c2 = np.where(ind_color == 0)[0]
                if len(ind_c2) == 0:
                    MAE_c2 = -1
                else:
                    MAE_c2 = MAE[ind_keep[0][ind_c2], ind_keep[1][ind_c2]]
                    MAE_c2 = np.mean(MAE_c2)
                    np.save(MAE_curv_folder + (3 - len(str(k))) * "0" + str(k) + "_zone1.npy", MAE_c2)

                ind_c3 = np.where(ind_color == 1)[0]
                if len(ind_c3) == 0:
                    MAE_c3 = -1
                else:
                    MAE_c3 = MAE[ind_keep[0][ind_c3], ind_keep[1][ind_c3]]
                    MAE_c3 = np.mean(MAE_c3)
                    np.save(MAE_curv_folder + (3 - len(str(k))) * "0" + str(k) + "_zone2.npy", MAE_c3)

                ind_c4 = np.where(ind_color == 2)[0]
                if len(ind_c4) == 0:
                    MAE_c4 = -1
                else:
                    MAE_c4 = MAE[ind_keep[0][ind_c4], ind_keep[1][ind_c4]]
                    MAE_c4 = np.mean(MAE_c4)
                    np.save(MAE_curv_folder + (3 - len(str(k))) * "0" + str(k) + "_zone3.npy", MAE_c4)


                geodesic_folder.append(geodesic_distance)
            all_geodesic.append(geodesic_folder)


def visibility():

    super_main_folder = "D:/PhD/Dropbox/CVPR_2024_2/results/"
    all_f = glob.glob(super_main_folder + "*")
    for main_folder in all_f :

        folder_GT = main_folder + "/GT/"

        # MESH GT OCTREE GENERATION

        pathObjGT2 = glob.glob(folder_GT + "mesh.ply")[0]
        mesh = trimesh.load(pathObjGT2)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces).astype(np.int32)

        camera_data = np.load(main_folder + "/cameras.npz")
        nb_views = len(camera_data.files) // 2
        print("before load")
        #rotation = trimesh.transformations.rotation_matrix(
        #    np.deg2rad(90), [1, 0, 0], point=[0, 0, 0])
        print("before octree")
        octree = ot.PyOctree(vertices, faces)
        count_vertices = np.zeros(vertices.shape[0])

        # Compute face normals
        face_normals = mesh.face_normals
        vertex_normals = np.zeros_like(mesh.vertices)
        for nf,face in tqdm(enumerate(mesh.faces)):
            vertex_normals[face] += face_normals[nf,:]
        vertex_normals = trimesh.unitize(vertex_normals).T

        for view in range(nb_views):

            mask = plt.imread(folder_GT+"/mask/"+"0"*(3-len(str(view)))+str(view)+".png")

            P_k = camera_data["world_mat_{}".format(view)]
            K, RT = load_K_Rt_from_P(P_k[:3, :])
            R = RT[:3,:3].T
            T = -RT[:3,:3].T @ RT[:3,[3]]
            K = K[:3,:3]

            origins = vertices.T
            cam_center = -R.T @ T

            dir = cam_center - origins
            ind_visible_cam = np.where((dir * vertex_normals).sum(axis=0) > 0)

            for nvert in tqdm(ind_visible_cam[0]):
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

"""

if __name__ == "__main__" :
    octree = None
    #visibility()
    #mae()
    mae_visibility()