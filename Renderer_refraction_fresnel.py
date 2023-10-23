import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

folder = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_RMVS_50_2/"
RR = np.load(folder+"R.npy")
TT = np.load(folder+"T.npy")
K = np.load(folder+"K.npy")

interface_data = np.load(folder+"interface.npz")
vertices = interface_data["vertices"].T
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
    pixels = np.concatenate((pixels[0].reshape(1,-1),pixels[1].reshape(1,-1),np.ones((1,len(pixels[0])))))

    rays_o = -R.T @ T
    dir = R.T @ (np.linalg.inv(K) @ pixels - T) - rays_o
    dir = dir / np.linalg.norm(dir,axis=0)
    rays_o = rays_o.reshape(3)

    idk = id_[kk]

    for k in tqdm(range(dir.shape[1])):
        ind = idk[int(pixels[0, k]), int(pixels[1, k])]
        dirk = dir[:,k]
        normal = normals[ind,:]

        ni = np.sum(normal*-dirk)
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

        image[int(pixels[0, k]), int(pixels[1, k]), :] = ni
        image2[int(pixels[0,k]),int(pixels[1,k]),:] = Ti * np.array([1.0,0.0,0.0]) + Ri
        image3[int(pixels[0, k]), int(pixels[1, k]), :3] = Ti * image3[int(pixels[0, k]), int(pixels[1, k]), :3] + Ri
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
    plt.imsave(folder+"images/"+(3-len(str(kk)))*"0"+str(kk)+".png",image3.clip(0.0,1.0))
    a=0


