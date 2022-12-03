##############################################
import os
import glob
import re

directory = r'G:\My Drive\SiPM_Project\LBS\LBS-APDvsSiPM\221122_AbsB_SiPM\\'
# print(directory)
path = r'G:\My Drive\SiPM_Project\LBS\LBS-APDvsSiPM\221122_AbsB_SiPM\*'

# print(path)
fileList = glob.glob(path, recursive=False)
print(fileList)
# #
for file in fileList:
    # os.rename(file, file+'.asc')
    # use if you want to see title which breast and chromophore is being plotted
    # file_name = os.path.splitext(os.path.basename(file)) #[0]
    file_name = os.path.basename(file) #[0]
    print(file_name)
    # my_path, ext = os.path.splitext(file)
    # print(my_path)
    regex = r"\d{2}\-\w+"
    if re.match(regex, file_name):
        print(file)
        newName = directory + file_name[3:9] +'-'+ file_name[0:2] +'-'+ file_name[10:]
        os.rename(file, newName)

    regex1 = r"\d{2}\w+\-"
    if re.match(regex1, file_name):
        # print(file)
        newName = directory + file_name[2:8] +'-'+ file_name[0:2] +'-'+ file_name[9:]
        # print(newName)
        os.rename(file, newName)
