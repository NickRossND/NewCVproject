# ##############################################
import os
import glob
import fnmatch
import re
import shutil
# # #
#################################
#
## this part of the code will move files from one location to another, specifically into a subdirectory within a directory
path = r'C:\Users\Nick\Downloads\PatientPlots\**\**\*.npy'
# print(path)
fileList = glob.glob(path, recursive=False)
# print(fileList)
## create list with strings to segment tumor from non-tumor data ##
tumors = ['DKU01*Right*', 'DKU02*Right*', 'DKU03*Left*', 'DKU05*Right*', 'DKU06*', 'DKU07*Right*', 'DKU08*Left*',
          'DKU09*Left*', 'DKU10*Right*', 'DKU11*Left*', 'DKU12*Left*', 'DKU13*Right*', 'DKU14*', 'DKU15*Right*',
          'DKU16*Right*', 'DKU18*Left*', 'DKU21*Left*', 'DKU22*', 'DKU24*Left*', 'DKU25*Left*', 'DKU26*Left*',
          'DKU27*Left*', 'DKU29*Left*', 'DKU30*Right*', 'DKU31*Left*', 'DKU32*Right*']
tumor_dir = r'C:\Users\Nick\Downloads\TumorStatus\Tumor\\'
notumor_dir = r'C:\Users\Nick\Downloads\TumorStatus\NoTumor\\'

for file in fileList:
    # print(file)
    ## use if you want to see title which breast and chromophore is being plotted
    # extract only the file name portion of the path
    # dirName =os.path.dirname(file)
    file_name = os.path.basename(file)
    # print(file_name)

    # if file name contains string of interest then rename/move to folder
    for tum in tumors:
        if fnmatch.fnmatch(file_name, tum): # use if looking to match file name with string
            # print(file_name)
            shutil.copy(file, tumor_dir)
        else:
            # print(file_name)
            shutil.copy(file, notumor_dir)

# path = r'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/**/*.png'
path2 = r'C:\Users\Nick\Downloads\TumorStatus\NoTumor\*.npy'
# # # print(path)
fileList2 = glob.glob(path2, recursive=True)
# print(fileList)
# tumors = ['DKU01*Right*', 'DKU02*Right*', 'DKU03*Left*', 'DKU05*Right*', 'DKU06*', 'DKU07*Right*', 'DKU08*Left*',
#           'DKU09*Left*', 'DKU10*Right*', 'DKU11*Left*', 'DKU12*Left*', 'DKU13*Right*', 'DKU14*', 'DKU15*Right*',
#           'DKU16*Right*', 'DKU18*Left*', 'DKU21*Left*', 'DKU22*', 'DKU24*Left*', 'DKU25*Left*', 'DKU26*Left*',
#           'DKU27*Left*', 'DKU29*Left*', 'DKU30*Right*', 'DKU31*Left*', 'DKU32*Right*']
for file2 in fileList2:
    file_name = os.path.basename(file2)
    for tum in tumors:
        if fnmatch.fnmatch(file_name, tum): # use if looking to match file name with string  os.remove(file)
            print(file_name)
            os.remove(file2)

## extra lines of code for quick coding/helpful items
# # extract only the file name portion of the path
#     file_name = os.path.splitext(os.path.basename(file))[0]
# # extract only the directory portion of the path
#     dirName =os.path.dirname(file)
#     print(file_name)
#     # my_path, ext = os.path.splitext(file)
#     # print(my_path)
#     # only for "Right" breast files
#     if fnmatch.fnmatch(file_name, '*NOISE*'):
#         print(file)
#         os.remove(file)
#     print('\n')
# #

###########################################
## used for renaming files ##
    # extract only the directory portion of the path
    # firstThree = file_name[0:3]
    # print(firstThree)
    # noise = r'\Noise\\'
    # print(dirName+noise+file_name)
    #     # print(dirName+noise+file_name)
    #     newFile = file_name[2:9]+file_name[0:2]+file_name[8:]
    #     newFile = (dirName + '\\' + newFile)
    #     os.rename(file, newFile)
    # else:
    #     newFile = file_name[3:9] + '-' + file_name[0:2]+file_name[9:]
    #     newFile = (dirName + '\\' + newFile)
    #     os.rename(file, newFile)
