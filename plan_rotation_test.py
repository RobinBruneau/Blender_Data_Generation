import numpy as np
from scipy.spatial.transform import Rotation


angles_x = np.array([0,-np.pi/2,0])
plan_x = Rotation.from_euler("xyz",angles_x,degrees=True).as_matrix()

nx = np.array([1,0,0])
ny = np.array([0,1,0])
nz = np.array([0,0,1])

ux = np.array([nx[1]/np.sqrt(nx[0]**2+nx[1]**2),-nx[0]/np.sqrt(nx[0]**2+nx[1]**2),0])
uy = np.array([ny[1]/np.sqrt(ny[0]**2+ny[1]**2),-ny[0]/np.sqrt(ny[0]**2+ny[1]**2),0])
uz = np.array([nz[1]/np.sqrt(nz[0]**2+nz[1]**2),-nz[0]/np.sqrt(nz[0]**2+nz[1]**2),0])

vx = np.cross(nx,ux)
vy = np.cross(ny,uy)
vz = np.cross(nz,uz)

Px = np.array([ux,vx,nx]).T
Py = np.array([uy,vy,ny]).T
Pz = np.array([uz,vz,nz]).T

angle_x = Rotation.from_matrix(Px).as_euler("xyz",degrees=True)
a=0
