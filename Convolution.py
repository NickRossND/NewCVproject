# script to add gaussian noise to data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp2d
import os
import glob
import fnmatch
import torch
from scipy import signal

## create new folders in patient folders ##
def newFolders(path):
    # print(path)
    folders = glob.glob(path, recursive=False)
    # print(folders)
    subDir = ["Interpolated","Noise", "Brightness","Contrast","Sharp_Blur"]
    for file in folders:
        for name in subDir:
            file_name = os.path.splitext(os.path.basename(file))[0]
            # print(file_name)
            newpath = os.path.join(file, name)
            if os.path.isdir(newpath) == False and file_name.startswith("DK"):
                os.mkdir(newpath)

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def bilinInterp(normalData):
    x = range(normalData.shape[0])
    y = range(normalData.shape[1])
    # print(x)
    # print(y)
    ## perform bilinear interpolation of the data ##
    while True:
        try:
            f = interp2d(x, y, normalData, kind="linear", bounds_error=False)
            break
        except ValueError:
            f = interp2d(y, x, normalData, kind="linear", bounds_error=False)
            break
    ## create x,y positions for interpolated data and turn matrix into 6x6 for CNN
    xnew = np.linspace(x[0],x[-1],7, endpoint=False)[1:]
    ynew = np.linspace(y[0],y[-1],7, endpoint=False)[1:]
    fnew = f(xnew, ynew)
    # print(xnew)
    # print(ynew)
    # print(fnew)
    # print("original data",original_data)
    # print('\n')
    # print("interpolated data",fnew)
    return fnew

dirpath = r'C:\Users\Nick\Downloads\PatientPlots\DKU01_140502'
newFolders(dirpath)

path = r'C:\Users\Nick\Downloads\PatientPlots\DKU01_140502\*.txt'
# path = r'G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\CVProject\PatientPlots\**\*.txt'
# print(path)
txtFiles = glob.glob(path, recursive=False)

##how much noise the system will generate
percentAmt = [1, 3, 5, 7, 10]

## set vars for changing contrast and brightness
alpha = [0.1, 0.25, 0.5, 0.75, 0.90]
beta = [0.1, 0.225, 0.35, 0.475, 0.6]



for file in txtFiles:
    ## Separate out different parts of the path name to create new file names later
    # file_name = os.path.splitext(os.path.basename(file))[0]
    print("curent folder", file)
    dirName =os.path.dirname(file)
    file_name = os.path.splitext(os.path.basename(file))[0]
    my_path, ext = os.path.splitext(file)
    print(file_name)

    sharpBlurFldr = '\\Sharp_Blur\\'
    filenameShBlr = dirName + sharpBlurFldr + file_name

    # read the data from text file
    original_data = np.genfromtxt(file, dtype=float)
    # only for "Right" breast files
    if fnmatch.fnmatch(file, '*Right*'):
        # print(file)
        ## horizontally flip the data so that it matches structure of original data
        original_data = np.flip(original_data, 1)

    # print("Matrix dimension: ", dimensions)
    ## normalize the data before interpolation ##
    norm_ogData = normalize_2d(original_data)
    interpData = bilinInterp(norm_ogData)
    print(interpData)
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    gausBlur = (1/8.0) * np.array([[1., 2., 1.],
                                    [2., 4., 2.],
                                    [1., 2., 1.]])
    ## Use if manually padding data ##
    # padded = np.pad(interpData, 1, mode='constant')
    # convData = signal.convolve2d(padded, sharpen, mode='valid')

    ## performs the same as the line of code above but does not need to pad the data before convolution ##
    sharpConv = signal.convolve2d(interpData, sharpen, boundary='fill', mode='same')
    blurConv = signal.convolve2d(interpData, gausBlur, boundary='fill', mode='same')

    sharpText = dirpath + '\Sharp_Blur\\' + file_name + '-Sharp'
    blurText = dirpath + '\Sharp_Blur\\' + file_name + '-Blur'
    np.save(sharpText, sharpConv)
    np.save(blurText, blurConv)
    print("Convolved:\n", sharpConv)
    print("Convolved2:\n", blurConv)

    # ## save the images ##
    # sns.set()
    # plt.figure(1)
    # sharp_fig = sns.heatmap(sharpConv, cmap='jet', cbar=False)
    # sharp_img = filenameShBlr + '-Sharp.png'
    # plt.savefig(sharp_img, bbox_inches='tight', pad_inches=0.001)
    #
    # plt.figure(2)
    # # plt.title('Noisy data: ' + file_name)
    # blur_fig = sns.heatmap(blurConv, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    # blur_img = filenameShBlr + '-Blur.png'
    # plt.savefig(blur_img, bbox_inches='tight', pad_inches=0.001)

    # plt.show()
