import numpy as np
import os

to_copy = "D:/PhD/Projects/Playing_with_NeuS/data/Graphosoma_MVPS/cameras.npz"

folder = "D:/PhD/Projects/Playing_with_NeuS/data/Data_Gaphosoma_MVPS_lowres/"
to_modify = folder + "cameras.npz"


data_copy = np.load(to_copy)
data_modify = np.load(to_modify)

data_finish = {}

for e in data_copy.files :
    if e[0] == "s" :
        data_finish.update({e:data_copy[e]})

for e in data_modify.files :
    data_finish.update({e:data_copy[e]})

del data_modify

os.rename(to_modify,folder+"camera_blend.npz")
np.savez(to_modify,**data_finish)
