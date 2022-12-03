# ##############################################
import os
import glob
import fnmatch
import re
# # #
# # # path = r'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/**/*.png'
# path = r'C:\Users\Nick\Downloads\PatientPlots\**\*.npy'
# # # print(path)
# fileList = glob.glob(path, recursive=True)
# # print(fileList)
#
# for file in fileList:
#     os.remove(file)
    # print(file)
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

#################################
# this part of the code will move files from one location to another, specifically into a subdirectory within a directory
path = r'C:\Users\Nick\Downloads\PatientPlots\**\Interpolated\*.npy'
# print(path)
fileList = glob.glob(path, recursive=False)
# print(fileList)
for file in fileList:
    # print(file)
    ## use if you want to see title which breast and chromophore is being plotted
    # extract only the file name portion of the path
    # dirName =os.path.dirname(file)
    file_name = os.path.basename(file)
    # print(file_name)
    # extract only the directory portion of the path
    # firstThree = file_name[0:3]
    # print(firstThree)
    # noise = r'\Noise\\'
    # print(dirName+noise+file_name)
    # if file name contains string of interest then rename/move to folder
    # if fnmatch.fnmatch(firstThree, '*n'): # use if looking to match file name with string
    #     # print(file)
    #     # print(dirName+noise+file_name)
    #     newFile = file_name[2:9]+file_name[0:2]+file_name[8:]
    #     newFile = (dirName + '\\' + newFile)
    #     os.rename(file, newFile)
    # else:
    #     newFile = file_name[3:9] + '-' + file_name[0:2]+file_name[9:]
    #     newFile = (dirName + '\\' + newFile)
    #     os.rename(file, newFile)

