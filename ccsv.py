import os
import csv
import glob
import natsort
import pandas as pd
path = 'amodal4'
file = "/media/bizon/Elements/pytorch-CycleGAN-and-pix2pix/weather1"

print(file)
'''
col_list = ["img_name","Val","Test", "Downtow"]
print(files)
df = pd.read_csv("tvt2.csv")

print(list(df.columns))

#print(files)
ls=[]
list=[]
for i in range(len(df["img_name"])):
    if df["Collegetown "][i]==1:
        ls.append(df["img_name"].astype(int)[i])


print(len(ls))
'''
#list=[52,54,56,58,60,62,64,66,68,70,72,74,76,78]

with open('td10.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    #for file in files:
    path1=natsort.natsorted(glob.glob(file+"/genA10/*.png"),reverse=False)

    path2=natsort.natsorted(glob.glob(file+"/"+"maskA/*.png"),reverse=False)
    path3=natsort.natsorted(glob.glob(file+"/"+"segA/*.png"),reverse=False)
    #path4=natsort.natsorted(glob.glob(file+"/"+"depth/*.png"),reverse=False)
    for i in range(len(path2)):
        tar=int(path2[i].split("/")[-1].split(".")[0])
        #if tar in ls:
        #temp=path1[i]+', '+path3[i]+', '+path2[i]
        writer.writerow([file+"/genA10/"+path2[i].split("/")[-1].split(".")[0]+".png",file+"/segA/"+path2[i].split("/")[-1],path2[i],file+"/depthA/"+path2[i].split("/")[-1].split(".")[0]+".npy"])
