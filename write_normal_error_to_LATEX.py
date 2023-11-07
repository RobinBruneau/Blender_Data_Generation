import glob
import os.path
import shutil
import subprocess
import numpy as np

os.system('cd ../../Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/ && git fetch && git pull')


def data_to_tab_latex(data,methods,names,table_id=0,caption="") :
    rows = len(data)
    cols = len(data[0])

    ind_Li = methods.index("Li\_19")

    data_fix = np.copy(data)
    data_fix[ind_Li, :] += 10000

    min_name = np.min(data_fix,axis=0,keepdims=True)
    first = (data_fix - min_name) < 1e-5



    second_name = np.min(data_fix+first*1000,axis=0,keepdims=True)
    second = (data_fix - second_name) < 1e-5

    average = np.mean(data,axis=1)
    average_fix = np.copy(average)
    average_fix[ind_Li] += 10000
    first_average = (average_fix - np.min(average_fix)) < 1e-5
    second_average = (average_fix - np.min(average_fix+first_average*1000)) < 1e-5

    latex_code = "\\begin{table*}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|c|" + cols * "c" + "|c|}\n"
    latex_code += "\\hline\n"
    latex_code += "& \\multicolumn{"+str(cols+1)+"}{c|}{Normal MAE $\\downarrow$}\\\\\n"

    latex_code += "Methods "
    for name in names :
        latex_code += "& "+name + " "
    latex_code += "& Average \\\\\n"
    latex_code += "\\hline\n"

    for k in range(rows):
        line = methods[k] +" & "
        for j in range(cols):
            if first[k,j] :
                line += " \\textcolor{cyan}{\\textbf{" + f"{data[k][j]:.3f}" + "}} &"
            else :
                if second[k,j] :
                    line += " \\textcolor{magenta}{\\underline{" + f"{data[k][j]:.3f}" + "}} &"
                else:
                    line += " "+f"{data[k][j]:.3f}"+ " &"
        if first_average[k]:
            line += " \\textcolor{cyan}{\\textbf{" + f"{average[k]:.3f}" + "}}\\\\\n"
        else :
            if second_average[k]:
                line += " \\textcolor{magenta}{\\underline{" + f"{average[k]:.3f}" + "}}\\\\\n"
            else :
                line += " "+f"{average[k]:.3f}"+"\\\\\n"
        latex_code+= line

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{"+caption+"}\n"
    latex_code += "\\label{table: "+str(table_id)+"}\n"
    latex_code += "\\end{table*}\n"

    return latex_code


def data_to_double_tab_latex(data1,data2,methods,names,table_id=0,caption="") :
    rows = len(data1)
    cols = len(data1[0])

    ind_Li = methods.index("Li\_19")

    # DATA 1
    data_fix1 = np.copy(data1)
    data_fix1[ind_Li, :] += 10000
    min_name1 = np.min(data_fix1,axis=0,keepdims=True)
    first1 = (data_fix1 - min_name1) < 1e-5
    second_name1 = np.min(data_fix1+first1*1000,axis=0,keepdims=True)
    second1 = (data_fix1 - second_name1) < 1e-5
    average1 = np.mean(data1,axis=1)
    average_fix1 = np.copy(average1)
    average_fix1[ind_Li] += 10000
    first_average1 = (average_fix1 - np.min(average_fix1)) < 1e-5
    second_average1 = (average_fix1 - np.min(average_fix1+first_average1*1000)) < 1e-5

    # DATA 2
    data_fix2 = np.copy(data2)
    data_fix2[ind_Li, :] += 10000
    min_name2 = np.min(data_fix2, axis=0, keepdims=True)
    first2 = (data_fix2 - min_name2) < 1e-5
    second_name2 = np.min(data_fix2 + first2 * 1000, axis=0, keepdims=True)
    second = (data_fix2 - second_name2) < 1e-5
    average2 = np.mean(data2, axis=1)
    average_fix2 = np.copy(average2)
    average_fix2[ind_Li] += 10000
    first_average2 = (average_fix2 - np.min(average_fix2)) < 1e-5
    second_average2 = (average_fix2 - np.min(average_fix2 + first_average2 * 1000)) < 1e-5

    latex_code = "\\begin{table*}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{adjustbox}{width=\\linewidth,center}\n"
    latex_code += "\\begin{tabular}{|c|" + cols * "c" + "|c||"++ cols * "c" + "|c|}\n"
    latex_code += "\\hline\n"
    latex_code += "& \\multicolumn{"+str(cols+1)+"}{c||}{Normal MAE (Global) $\\downarrow$} & \\multicolumn{"+str(cols+1)+"}{c|}{Normal MAE (Low Visibility) $\\downarrow$}\\\\\n"

    latex_code += "Methods "
    for name in names :
        latex_code += "& "+name + " "
    latex_code += "& Average"
    for name in names :
        latex_code += "& "+name + " "
    latex_code += "& Average \\\\\n"
    latex_code += "\\hline\n"

    for k in range(rows):
        line = methods[k] +" & "

        #data 1
        for j in range(cols):
            if first1[k,j] :
                line += " \\textcolor{cyan}{\\textbf{" + f"{data1[k][j]:.3f}" + "}} &"
            else :
                if second[k,j] :
                    line += " \\textcolor{magenta}{\\underline{" + f"{data1[k][j]:.3f}" + "}} &"
                else:
                    line += " "+f"{data1[k][j]:.3f}"+ " &"
        if first_average1[k]:
            line += " \\textcolor{cyan}{\\textbf{" + f"{average1[k]:.3f}" + "}} &"
        else :
            if second_average1[k]:
                line += " \\textcolor{magenta}{\\underline{" + f"{average1[k]:.3f}" + "}} &"
            else :
                line += " "+f"{average1[k]:.3f}"+" &"

        #data 2
        for j in range(cols):
            if first2[k,j] :
                line += " \\textcolor{cyan}{\\textbf{" + f"{data2[k][j]:.3f}" + "}} &"
            else :
                if second[k,j] :
                    line += " \\textcolor{magenta}{\\underline{" + f"{data2[k][j]:.3f}" + "}} &"
                else:
                    line += " "+f"{data2[k][j]:.3f}"+ " &"
        if first_average2[k]:
            line += " \\textcolor{cyan}{\\textbf{" + f"{average2[k]:.3f}" + "}}\\\\\n"
        else :
            if second_average2[k]:
                line += " \\textcolor{magenta}{\\underline{" + f"{average2[k]:.3f}" + "}}\\\\\n"
            else :
                line += " "+f"{average2[k]:.3f}"+"\\\\\n"
        latex_code+= line

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{adjustbox}\n"
    latex_code += "\\caption{"+caption+"}\n"
    latex_code += "\\label{table: "+str(table_id)+"}\n"
    latex_code += "\\end{table*}\n"

    return latex_code



def data_to_im_tab_latex(data,legende,fig_id,caption,ratio="width",ratio_size=0.12) :

    rows = len(data)
    cols = len(data[0])




    latex_code = "\\begin{figure*}[!ht]\n" \
                 "\centering\n" \
                 "\\begin{tabular}{" + cols * "c" + "}\n"


    for k in range(rows):
        for j in range(cols):

            latex_code += "\includegraphics["+ratio+" = " + str(ratio_size) + "\linewidth]{"+data[k][j]+"}"
            if j != cols - 1:
                latex_code += " &\n"
            else:
                latex_code += "\\\\\n"

    if len(legende) == cols:
        for k, leg in enumerate(legende):
            if k != len(legende) - 1:
                latex_code += leg + " & "
            else:
                latex_code += leg + "\n"

    latex_code += "\end{tabular}\n" +\
                  "\caption{"+caption+"}\n"+ \
                  "\label{fig:" + str(fig_id) + "}\n" \
                  "\end{figure*}"

    return latex_code

names_view = {
              'bearPNG':0,
              'buddhaPNG':0,
              'cowPNG':0,
              'pot2PNG':0,
              'readingPNG':0
              }


main_folder = "D:/PhD/Dropbox/CVPR_2024/check_normals_and_visibility/"
all_methods = ["result_Park_16","result_Li_19","result_PS_NeRF_22","result_Kaya_22","result_Kaya_23","result_MVPSNet_23","input_SDM",
               "result_Ours_old","result_Ours_(a)","result_Ours_(b)"]
all_methods_names = ["Park\_16","Li\_19","PS\_NeRF\_22","Kaya\_22","Kaya\_23","MVPSNet\_23","SDM",
                     "Ours\_old","Ours\_(a)","Ours\_(b)"]

all_names = []
for name in names_view.keys():
    if names_view[name] != -1 :
        all_names.append(name[:-3].capitalize())


fig_id = 0

fig_folder = "D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/figures/figure_{}".format(fig_id)
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)
else :
    files = glob.glob(fig_folder+"/*")
    for elem in files :
        os.remove(elem)

data = []
for key in names_view.keys():
    row = []
    if names_view[name] != -1 :
        for method in all_methods :
            folder = main_folder + "/"+key+"/"+method+"/"
            all_im = glob.glob(folder+"/geo*crop/"+(3-len(str(names_view[key])))*"0"+str(names_view[key])+".jpg")
            for i,im in enumerate(all_im):
                im = im.replace("\\", "/")
                shutil.copy(im, fig_folder + "/" + key + "_" + method + "_" + str(names_view[key]) + ".jpg")
                row.append("./figures/"+"figure_{}".format(fig_id)+"/"+key + "_" + method + "_" + str(names_view[key]) + ".jpg")


        data.append(row)


data_MAE = []
for key in names_view.keys():
    row_MAE = []
    for method in all_methods :
        mae_path = glob.glob(main_folder+"/"+key+"/"+method+"/MAE/*.npy")
        MAE_values = [np.load(mp) for mp in mae_path]
        MAE = np.mean(MAE_values)
        row_MAE.append(MAE * 180 / np.pi)
    data_MAE.append(row_MAE)

data_MAE_vis_z1 = []
data_MAE_vis_z2 = []
data_MAE_vis_z3 = []
data_MAE_vis_z4 = []
for key in names_view.keys():
    row_MAE_z1 = []
    row_MAE_z2 = []
    row_MAE_z3 = []
    row_MAE_z4 = []
    for method in all_methods :
        mae_path_z1 = glob.glob(main_folder+"/"+key+"/"+method+"/MAE_vis/*zone1.npy")
        MAE_values_z1 = [np.load(mp) for mp in mae_path_z1]
        MAE_z1 = np.mean(MAE_values_z1)* 180 / np.pi

        mae_path_z2 = glob.glob(main_folder + "/" + key + "/" + method + "/MAE_vis/*zone2.npy")
        MAE_values_z2 = [np.load(mp) for mp in mae_path_z2]
        MAE_z2 = np.mean(MAE_values_z2) * 180 / np.pi

        mae_path_z3 = glob.glob(main_folder + "/" + key + "/" + method + "/MAE_vis/*zone3.npy")
        MAE_values_z3 = [np.load(mp) for mp in mae_path_z3]
        MAE_z3 = np.mean(MAE_values_z3) * 180 / np.pi

        mae_path_z4 = glob.glob(main_folder + "/" + key + "/" + method + "/MAE_vis/*zone4.npy")
        MAE_values_z4 = [np.load(mp) for mp in mae_path_z4]
        MAE_z4 = np.mean(MAE_values_z4) * 180 / np.pi

        row_MAE_z1.append(MAE_z1)
        row_MAE_z2.append(MAE_z2)
        row_MAE_z3.append(MAE_z3)
        row_MAE_z4.append(MAE_z4)
    data_MAE_vis_z1.append(row_MAE_z1)
    data_MAE_vis_z2.append(row_MAE_z2)
    data_MAE_vis_z3.append(row_MAE_z3)
    data_MAE_vis_z4.append(row_MAE_z4)


data_MAE_curv_z1 = []
data_MAE_curv_z2 = []
data_MAE_curv_z3 = []
for key in names_view.keys():
    row_MAE_z1 = []
    row_MAE_z2 = []
    row_MAE_z3 = []
    for method in all_methods :
        mae_path_z1 = glob.glob(main_folder+"/"+key+"/"+method+"/MAE_curv/*zone1.npy")
        MAE_values_z1 = [np.load(mp) for mp in mae_path_z1]
        MAE_z1 = np.mean(MAE_values_z1)* 180 / np.pi

        mae_path_z2 = glob.glob(main_folder + "/" + key + "/" + method + "/MAE_curv/*zone2.npy")
        MAE_values_z2 = [np.load(mp) for mp in mae_path_z2]
        MAE_z2 = np.mean(MAE_values_z2) * 180 / np.pi

        mae_path_z3 = glob.glob(main_folder + "/" + key + "/" + method + "/MAE_curv/*zone3.npy")
        MAE_values_z3 = [np.load(mp) for mp in mae_path_z3]
        MAE_z3 = np.mean(MAE_values_z3) * 180 / np.pi

        row_MAE_z1.append(MAE_z1)
        row_MAE_z2.append(MAE_z2)
        row_MAE_z3.append(MAE_z3)
    data_MAE_curv_z1.append(row_MAE_z1)
    data_MAE_curv_z2.append(row_MAE_z2)
    data_MAE_curv_z3.append(row_MAE_z3)


data_MAE = np.array(data_MAE).transpose()
data_MAE_vis_z1 = np.array(data_MAE_vis_z1).transpose()
data_MAE_vis_z2 = np.array(data_MAE_vis_z2).transpose()
data_MAE_vis_z3 = np.array(data_MAE_vis_z3).transpose()
data_MAE_vis_z4 = np.array(data_MAE_vis_z4).transpose()
data_MAE_curv_z1 = np.array(data_MAE_curv_z1).transpose()
data_MAE_curv_z2 = np.array(data_MAE_curv_z2).transpose()
data_MAE_curv_z3 = np.array(data_MAE_curv_z3).transpose()
latex_code = data_to_im_tab_latex(data,all_methods_names,fig_id=0,caption="this is a caption",ratio="width",ratio_size=0.07)
latex_code_2 = data_to_tab_latex(data_MAE,all_methods_names,all_names,table_id=fig_id,caption="Everything")
latex_code_3 = data_to_tab_latex(data_MAE_vis_z1,all_methods_names,all_names,table_id=fig_id+1,caption="Visibility 1-5")
latex_code_4 = data_to_tab_latex(data_MAE_vis_z2,all_methods_names,all_names,table_id=fig_id+2,caption="Visibility 6-10")
latex_code_5 = data_to_tab_latex(data_MAE_vis_z3,all_methods_names,all_names,table_id=fig_id+3,caption="Visibility 11-15")
latex_code_6 = data_to_tab_latex(data_MAE_vis_z4,all_methods_names,all_names,table_id=fig_id+4,caption="Visibility 16-20")

latex_code_7 = data_to_tab_latex(data_MAE_curv_z1,all_methods_names,all_names,table_id=fig_id+5,caption="Curvature : CONCAVITY")
latex_code_8 = data_to_tab_latex(data_MAE_curv_z2,all_methods_names,all_names,table_id=fig_id+6,caption="Curvature : OTHERS")
latex_code_9 = data_to_tab_latex(data_MAE_curv_z3,all_methods_names,all_names,table_id=fig_id+7,caption="Curvature : CONVEXITY")

latex_code_10 = data_to_double_tab_latex(data_MAE,data_MAE_vis_z1,all_methods_names,all_names,table_id=fig_id+8,caption="MAE : Gobal \& Low Visibility (1-5)")



f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/figures/figure_{}.tex".format(fig_id),'w')
f.write(latex_code)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id),'w')
f.write(latex_code_2)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+1),'w')
f.write(latex_code_3)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+2),'w')
f.write(latex_code_4)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+3),'w')
f.write(latex_code_5)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+4),'w')
f.write(latex_code_6)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+5),'w')
f.write(latex_code_7)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+6),'w')
f.write(latex_code_8)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+7),'w')
f.write(latex_code_9)
f.close()

f=open("D:/PhD/Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/tables/table_{}.tex".format(fig_id+8),'w')
f.write(latex_code_10)
f.close()

os.system('cd ../../Dropbox/CVPR_2024/654a00c0b5050395ccf785b7/ && git add * && git commit -m "maj" && git push')

