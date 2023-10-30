import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import cv2


def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    t_near = -np.inf
    t_far = np.inf
    intersect_face = None

    for i in range(3):
        # Check for parallel rays
        if abs(ray_direction[i]) < 1e-6:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return False,None, None  # No intersection, no normal
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_direction[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_direction[i]
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))

    if t_near > t_far or t_far < 0:
        return False,None, None  # No intersection, no normal

    intersection_point = ray_origin + t_near * ray_direction

    for i in range(3):
        if t_near == (box_min[i] - ray_origin[i]) / ray_direction[i]:
            normal = np.zeros(3)
            normal[i] = -1
            return True,intersection_point, normal

        if t_near == (box_max[i] - ray_origin[i]) / ray_direction[i]:
            normal = np.zeros(3)
            normal[i] = 1
            return True,intersection_point, normal

    return False,intersection_point, None


folder = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_RMVS_50_2/"
RR = np.load(folder+"R.npy")
TT = np.load(folder+"T.npy")
K = np.load(folder+"K.npy")

interface_data = np.load(folder+"interface.npz")
vertices = interface_data["vertices"].T
box_min = np.min(vertices,axis=0)
box_max = np.max(vertices,axis=0)
centers = interface_data["centers"].T
faces = (interface_data["faces"].T).astype(int)
normals = (interface_data["normals"] / np.linalg.norm(interface_data["normals"],axis=0)).T
d = []
for k in range(faces.shape[0]):
    a = normals[k,:]
    b = vertices[faces[k,0],:]
    d.append(-np.sum(a*b))
d = np.array(d)

maskp = glob.glob(folder+"medium_masks/*")


imagesp = glob.glob(folder+"images_with/*")
all_images = []
for imp in imagesp :
    all_images.append(plt.imread(imp))
a=0

idp = glob.glob(folder+"interface/*.png")

masks = []
for mp in maskp :
    masks.append(plt.imread(mp)[:,:,0])
a=0

id_ = []
for i in idp :
    id_.append((plt.imread(i)[:,:,0]*255.0).astype(int))
a=0


all_im_id = []
for k in range(RR.shape[2]):
    R = RR[:,:,k].reshape(3,3)
    T = TT[:,[k]]
    cam_pos = (-R.T @ T).reshape(3)
    im = np.zeros((1080, 1920,3))

    for kk in range(faces.shape[0]):

        center = np.mean(vertices[faces[kk, :], :], axis=1)
        if ((center - cam_pos).dot(normals[kk, :]) < 0):
            vert = vertices[faces[kk, :], :].T
            vert_cam = K @ (R@vert+T)
            vert_cam /= vert_cam[2,:]
            vert_pixel = np.round(vert_cam[:2,:]).astype(int)
            pt1 = (vert_pixel[0,0],vert_pixel[1,0])
            pt2 = (vert_pixel[0,1],vert_pixel[1,1])
            pt3 = (vert_pixel[0,2],vert_pixel[1,2])
            triangle_cnt = np.array([pt1, pt2, pt3])
            color = (kk/255.0,kk/255.0,kk/255.0)
            cv2.drawContours(im, [triangle_cnt], contourIdx=0, color=color, thickness=cv2.FILLED)

    all_im_id.append(np.round(im[:,:,0]*255.0).astype(int))

a=0
for kk in range(len(masks)):

    image = np.copy(masks[kk]).reshape(masks[0].shape[0],masks[0].shape[1],1)
    image[0,0] = 1.0
    image2 = np.concatenate((image,image,image),axis=2)
    image = np.concatenate((image, image, image), axis=2)
    image3 = all_images[kk]
    image4 = np.copy(all_images[kk])
    R = RR[:,:,kk].reshape(3,3)
    T = TT[:,[kk]]
    mask = masks[kk]
    pixels = np.where(mask > 0.5)
    pixels = np.concatenate((pixels[1].reshape(1,-1),pixels[0].reshape(1,-1),np.ones((1,len(pixels[0])))))

    rays_o = -R.T @ T
    dir = R.T @ (np.linalg.inv(K) @ pixels - T) - rays_o
    dir = dir / np.linalg.norm(dir,axis=0)
    rays_o = rays_o.reshape(3)

    idk = id_[kk]

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter(rays_o[0],rays_o[1],rays_o[2],color='b')
    #mult = 2.0
    #for nray in range(dir.shape[1]//10000):
    #    ax.plot([rays_o[0],rays_o[0]+mult*dir[0,500*nray]],
    #            [rays_o[1],rays_o[1]+mult*dir[1,500*nray]],
    #            [rays_o[2],rays_o[2]+mult*dir[2,500*nray]])
    #ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],color='r')
    #plt.show()



    for k in tqdm(range(dir.shape[1])):

        ind = all_im_id[kk][int(pixels[1, k]), int(pixels[0, k])]
        dirk = dir[:,k]
        #normal = normals[ind,:]

        is_intersect,intersection_,normal = ray_box_intersection(ray_origin=rays_o,ray_direction=dirk,box_min=box_min,box_max=box_max)
        a=0
        if is_intersect :
            ni = np.sum(normal*-dirk)
            a=0
            if ni >= 0 :
                mu = 1.0 / 1.56
                is_total_reflection = (ni * ni < 1 - 1 / (mu * mu));
                if is_total_reflection :
                    Ri = 1.0
                    Ti = 0.0
                else :
                    Rs = (mu * ni - np.sqrt(1-(mu * mu) * (1-ni * ni)))**2 /((mu * ni + np.sqrt(1-(mu * mu) * (1-ni * ni))))**2
                    Re = (mu * np.sqrt(1-(mu * mu) * (1-ni * ni)) - ni )**2 /((mu * np.sqrt(1-(mu * mu) * (1-ni * ni)) + ni ))**2
                    Ri = 0.5 * (Rs+Re)
                    Ti = 1-Ri
                a=0

                image[int(pixels[1, k]), int(pixels[0, k]), :] = ni
                image2[int(pixels[1,k]),int(pixels[0,k]),:] = Ti * np.array([1.0,0.0,0.0]) + Ri

                color =  Ti * image3[int(pixels[1, k]), int(pixels[0, k]), :3] + Ri
                min_color = np.min(color)
                max_color = np.max(color)
                assert(min_color >= 0.0)
                assert(min_color <= 1.0)
                assert(max_color >= 0.0)
                assert(max_color <= 1.0)
                image3[int(pixels[1, k]), int(pixels[0, k]), :3] = color
                #image3[int(pixels[0, k]), int(pixels[1, k]), :3] = (image3[int(pixels[0, k]), int(pixels[1, k]), :3] - Ri)/Ti
                a=0

    #plt.figure()
    #plt.subplot(2,2,1)
    #plt.imshow(image)
    #plt.subplot(2, 2, 2)
    #plt.imshow(image2)
    #plt.subplot(2, 2, 3)
    #plt.imshow(image4)
    #plt.subplot(2, 2, 4)
    #plt.imshow(image3)
    #plt.show()
    plt.imsave(folder+"images_back_to_no_fresnel/"+(3-len(str(kk)))*"0"+str(kk)+".png",image3)
    a=0


