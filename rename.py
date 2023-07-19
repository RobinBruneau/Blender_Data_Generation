import os
import glob

output_path = "D:/PhD/Projects/Playing_with_NeuS/data/MVPS_sphere_normal/"

im = glob.glob(output_path+"normal/*")
for i in im :
    before= i.split("\\")[0]
    name = i.split("\\")[-1].split("_cut_")[0]
    b = before+"\\"+name+".png"
    os.rename(i,before+"\\"+name+".png")


