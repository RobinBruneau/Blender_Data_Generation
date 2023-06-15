import numpy as np

def generate_local_rotation(u,min_angle,max_angle):
    k = np.random.rand(3)
    x = k - np.dot(k, u) * u
    x /= np.linalg.norm(x)
    y = np.cross(u, x)

    P = np.ones((3, 3))
    P[:, 0] = u
    P[:, 1] = x
    P[:, 2] = y

    u_cone = np.cos(np.pi * max_angle / 180)
    pts = np.random.rand(1, 2) * 2 - 1.0
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= np.sin(np.pi * min_angle / 180) + np.random.rand(1, 1) * (np.sin(np.pi * max_angle / 180)-np.sin(np.pi * min_angle / 180))
    x_cone = pts[:, 0]
    y_cone = pts[:, 1]

    pt_cone = np.ones((3, 1))
    pt_cone[0, :] = u_cone
    pt_cone[1, :] = x_cone
    pt_cone[2, :] = y_cone

    u_rot = P @ pt_cone
    u_rot /= np.linalg.norm(u_rot)

    return u_rot.reshape(3)



folder = "D:\PhD\Projects\Playing_with_NeuS\data\MVPS_graphosoma_testing_raw_normal/"

file = folder + "lights.npz"

data = np.load(file)
data2 = {}
for e in data.files :
    l = data[e]
    l2 = np.zeros(l.shape)
    for k,u in enumerate(l) :
        u_rot = generate_local_rotation(u,3,7)
        l2[k] = u_rot

    angles_ = np.arccos((l * l2).sum(axis=1))*180/np.pi
    data2.update({e:l2})

np.savez(folder+"lights_noise.npz",**data2)
a=0