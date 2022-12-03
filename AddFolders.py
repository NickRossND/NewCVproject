##############################################
import os
import glob
import fnmatch

path = r'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/**/'
# print(path)
folders = glob.glob(path, recursive=False)
print(folders)

for file in folders:
    # file_name = os.path.splitext(os.path.basename(file))[0]
    # print(file_name)
    newDir = "Noise"
    newpath = os.path.join(file, newDir)
    os.mkdir(newpath)
