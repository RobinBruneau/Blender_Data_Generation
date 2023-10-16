import numpy as np

file_path = "D:/PhD/Projects/Playing_with_NeuS2/tools/data/neus/nails_normal/transform_train_base.obj"

f=open(file_path,"r")

vertices = []

for line in f :
    data = line.split(" ")
    if data[0] == "v" :
        vertices.append([data[1],data[2],data[3]])

vertices = np.array(vertices).astype(float)
ind = np.where(vertices < -0.5)
vertices[ind] = 0.0
min_bb = np.min(vertices,axis=0)
max_bb = np.max(vertices,axis=0)
center = (min_bb+max_bb)/2.0

# Actuellement S = 50 T = [-0.75*50 , 0.75*50 , 0]

center_world = center * 50 + np.array([-0.75*50 , 0.75*50 , 0])


# World = S * Neus + T
# Neus = (World-T)/S

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,3,1)
plt.hist(vertices[:,0])
plt.subplot(1,3,2)
plt.hist(vertices[:,1])
plt.subplot(1,3,3)
plt.hist(vertices[:,2])

plt.show()
a=0