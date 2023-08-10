import os
import glob

output_path = "/home/robin/Desktop/PhD/Data/Data_ortho_normal/"

im = glob.glob(output_path+"normal/*")
for k,i in enumerate(im) :
    before= i.split("/normal/")[0]
    name = i.split("/")[-1].split("_cut_")[0]
    b = before+"/"+name+".png"
    #os.rename(i,before+"/mask_medium/"+name+".png")
    os.rename(i, output_path+"/normal/"+(3-len(str(k)))*"0"+str(k)+".png")


