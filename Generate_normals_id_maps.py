import numpy as np
import cv2

folder = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_RMVS_50_2/"
RR = np.load(folder+"R.npy")
TT = np.load(folder+"T.npy")
K = np.load(folder+"K.npy")

interface_data = np.load(folder+"interface.npz")
vertices = interface_data["vertices"].T
faces = (interface_data["faces"].T).astype(int)
normals = (interface_data["normals"] / np.linalg.norm(interface_data["normals"],axis=0)).T
centers = interface_data["centers"].T
d = []
for k in range(faces.shape[0]):
    a = normals[k,:]
    b = vertices[faces[k,0],:]
    d.append(-np.sum(a*b))
d = np.array(d)


all_im = []
for k in range(RR.shape[2]):
    R = RR[:,:,k].reshape(3,3)
    T = TT[:,[k]]
    cam_pos = (-R.T @ T).reshape(3)
    im = np.zeros((1080, 1920,3))

    for kk in range(faces.shape[0]):

        center = np.mean(vertices[faces[kk, :], :],axis=1)
        if ((center-cam_pos).dot(normals[kk,:])<0):
            vert = vertices[np.concatenate((faces[kk, :],np.zeros(1).astype(int))), :].T
            vert_cam = K @ (R@vert+T)
            vert_cam /= vert_cam[2,:]
            vert_pixel = np.round(vert_cam[:2,:]).astype(int)
            pt1 = (vert_pixel[0,0],vert_pixel[1,0])
            pt2 = (vert_pixel[0,1],vert_pixel[1,1])
            pt3 = (vert_pixel[0,2],vert_pixel[1,2])
            triangle_cnt = np.array([pt1, pt2, pt3])
            color = (10*kk/255,10*kk/255,10*kk/255)
            print(color)
            print(centers[kk,:])
            print(vert)
            print(normals[kk, :])
            print("")
            cv2.drawContours(im, [triangle_cnt], contourIdx=0, color=color, thickness=cv2.FILLED)

    all_im_id.append(im[:,:,0])