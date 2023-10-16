import numpy as np
import glob


folder = "D:/PhD/Dropbox/Data/Data/Nails/calibration/"
subfolders = glob.glob(folder+"*")

s=20
tx = -28.22
ty = 50.768 + 10
tz = 9.88175
S = np.array([[s,0,0,tx],[0,s,0,ty],[0,0,s,tz],[0,0,0,1]])
data = {}

for k,sf in enumerate(subfolders) :
    RT = np.loadtxt(sf + "/world_to_camera.csv", delimiter=',')[:3, :]
    K = np.loadtxt(sf + "/camera_intrinsic.csv", delimiter=',')
    P = K @ RT
    P = np.concatenate((P,np.array([[0,0,0,1]])),axis=0)
    data.update({"world_mat_{}".format(k): P})
    data.update({"scale_mat_{}".format(k): S})

#np.savez("D:/PhD/Dropbox/Data/Data/Nails/NeuS/nails/cameras.npz", **data)
#np.savez("D:/PhD/Projects/Playing_with_NeuS2/data/neus_to_neus2/nails_normal_nail/cameras.npz", **data)

if True :

    file_path = "D:/PhD/Projects/Playing_with_NeuS2/tools/data/neus/nails_normal_nail/transform_train_base.obj"
    file_path2 = "D:/PhD/Projects/Playing_with_NeuS2/tools/data/neus/nails_normal_nail/transform_world.obj"
    f = open(file_path, "r")
    f2= open(file_path2,'w')
    vertices = []

    for line in f:
        data = line.split(" ")
        if data[0] == "v":
            vertice = (np.array([data[1],data[2],data[3]]).astype(float)*2.0-1.0) * s + np.array([tx,ty,tz])
            f2.write("v {} {} {}\n".format(vertice[0],vertice[1],vertice[2]))
        else :
            f2.write(line)

    f.close()
    f2.close()
