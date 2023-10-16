import numpy as np
import json


file = "D:/PhD/Dropbox/Data/Data/CHEVAL/cameras.sfm"

f=open(file,"rb")
data = json.load(f)
f.close()

views = data["views"]
intrinsics = data["intrinsics"][0]
poses = data["poses"]

width = float(intrinsics["width"])
height = float(intrinsics["height"])
sensorWidth = float(intrinsics["sensorWidth"])
focal = float(intrinsics["focalLength"])
f = focal * width / sensorWidth
x0 = width/2
y0 = height/2
K = np.array([[f,0,x0],[0,f,y0],[0,0,1]])

s=2
S = np.array([[s,0,0,0],[0,s,0,0],[0,0,s,1.5],[0,0,0,1]])

data = {}
for k,v in enumerate(views) :
    id = v["poseId"]
    print(v["path"])
    for p in poses :
        if p["poseId"] == id :
            Rt = np.array(p["pose"]["transform"]["rotation"]).reshape(3,3).astype(float)
            c = np.array(p["pose"]["transform"]["center"]).reshape(3,1).astype(float)
            R = Rt.T
            T = -Rt.T @ c
            P =  K @ np.concatenate((R,T),axis=1)
            a=0
            data.update({"world_mat_{}".format(k): P})
            data.update({"scale_mat_{}".format(k): S})

np.savez("D:/PhD/Dropbox/Data/Data/CHEVAL/cameras.npz", **data)
np.savez("D:/PhD/Projects/Playing_with_NeuS2/data/neus_to_neus2/Cheval/cameras.npz", **data)

