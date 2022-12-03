# script to add gaussian noise to data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import fnmatch
import torch
import time
from scipy.interpolate import interp2d
from scipy import signal

programTime = time.time()
### create new data with the same dimensions (7x7 or whatever) using bilinear or cubic interpolation
### normalize the data in some fashion (0-1 values) building multi-channel NN
### calc min and max for each database (training and testing) per category (each 7)
### then add noise, brightness, contrast to the data staying within 0-1

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
    return(file_name)
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
# define a path where the image is
dirpath = r'C:\Users\Nick\Downloads\PatientPlots\**'
# dirpath = r'G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\CVProjectnew\PatientPlots\**'
newFolders(dirpath)

path = r'C:\Users\Nick\Downloads\PatientPlots\**\*.txt'
# path = r'G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\CVProject\PatientPlots\**\*.txt'
# print(path)
txtFiles = glob.glob(path, recursive=False)

## how much noise the system will generate ranging from 0.01% to 1% ##
percentAmt = [1, 3, 5, 7, 10]

## set vars for changing contrast and brightness ##
alpha = [0.1, 0.25, 0.5, 0.75, 0.90]
beta = [0.1, 0.225, 0.35, 0.475, 0.6]

## sharpen and gaussian blur array to be applied after changing contrast to images ##
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
gausBlur = (1/8.0) * np.array([[1., 2., 1.],
                                [2., 4., 2.],
                                [1., 2., 1.]])
#
for file in txtFiles:
    ## Separate out different parts of the path name to create new file names later
    # file_name = os.path.splitext(os.path.basename(file))[0]
    startTime = time.time()
    print("curent folder", file)
    dirName =os.path.dirname(file)
    wantedName = os.path.split(dirName)[1] + '-'
    file_name = os.path.splitext(os.path.basename(file))[0]
    my_path, ext = os.path.splitext(file)
#     print("dirName: ", dirName)
#     print("wantedName: ", wantedName)
#     print("file_name: ", file_name)
#     print("my_path: ", my_path)
#
    ### used to create new folders in each patient folder ###
    interpFldr = '\\Interpolated\\'
    filenameInterp = dirName + interpFldr + wantedName + file_name

    nseFldr = '\\Noise\\'
    filenameNoise = dirName + nseFldr + wantedName + file_name

    betaFldr = '\\Brightness\\'
    filenameBeta = dirName + betaFldr + wantedName + file_name

    contFldr = '\\Contrast\\'
    filenameCont = dirName + contFldr + wantedName + file_name
    # fldrnameCont = dirName + contFldr

    sharpBlurFldr = '\\Sharp_Blur\\'
    filenameShBlr = dirName + sharpBlurFldr + wantedName + file_name
    # print(filenameShBlr)
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
    dimensions = interpData.shape
    np.save(filenameInterp+'-INTERP', interpData)
    # interp_fig = sns.heatmap(interpData, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    # interp_img = filenameInterp + '-INTERP.jpg'
    # plt.savefig(interp_img, bbox_inches='tight', pad_inches=0.001)
    # print("Data normalized and interpolated")

    ## create the noise ##
    mean = 0
    for x in percentAmt:
        ## set up gaussian noise to be added to the data
        var = 0.01*x
        sigma = np.multiply(interpData, var)
        #print(sigma)
        noise = np.random.normal(loc=mean,
                                 scale=sigma,
                                 size=dimensions)
        # print("This is noise:\n", noise)

        ## create new matrix with noise added to original data
        noisy_data = interpData + noise
        noisy_txt = filenameNoise + '-NOISE' + (str(int(var * 100)))
        # print("Noise:\n", noisy_data)
        ### Check if there are already files with noise in the name and create new files/imgs if they don't exist ##
        # if "NOISE" or "BRT" or "CONTRAST" or "Sharp" or "Blur" not in file_name:
        np.save(noisy_txt, noisy_data)
        #     # ## use seaborn library for heatmap to confirm that the transformation worked
        #     sns.set()
        #     plt.figure(1)
        #     og = sns.heatmap(original_data, cmap='jet', cbar=False)
        #     plt.figure(2)
        #     # plt.title('Noisy data: ' + file_name)
        #     noisy_fig = sns.heatmap(noisy_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
        #     nsy_img = filenameNoise + '-NOISE'+(str(int(var*100)))+'.jpg'
        #     plt.savefig(nsy_img, bbox_inches='tight', pad_inches=0.001)
        #     print("new file: ", noisy_txt)
        newNoisy = torch.tensor(noisy_data)

        ## run through each noise file and add brightness to each image ##
        for y in beta:
            temp = torch.tensor(noisy_data + y)
            temp2 = torch.ones(noisy_data.shape)
            beta_data = torch.where(temp > 1, temp2, temp)
            # print("with beta\n", beta_data )
            nseNbeta = '-NOISE'+(str(int(var*100))) + '-BRT' + str(int(y * 100))
            beta_txt = filenameBeta + nseNbeta
            # print(beta_txt)
            np.save(beta_txt, beta_data)
            # beta_fig = sns.heatmap(beta_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
            # beta_img = filenameBeta + nseNbeta +'.jpg'
            # plt.savefig(beta_img, bbox_inches='tight', pad_inches=0.001)

            ## Add contrast to each image that was just brightened ##
            for z in alpha:
                temp = beta_data.detach().cpu()
                temp = temp * z
                temp2 = torch.ones(beta_data.shape)
                cont_data = torch.where(temp > 1, temp2, temp)
                # print(cont_data)
                nseBetaCont = '-NOISE'+(str(int(var*100))) + '-BRT' + str(int(y * 100)) +'-CONT' + str(int(z * 100))
                cont_txt = filenameCont + nseBetaCont
                np.save(cont_txt, cont_data.numpy())
                # cont_img = filenameCont + nseBetaCont + '.jpg'
                # cont_fig = sns.heatmap(cont_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
                # plt.savefig(cont_img, bbox_inches='tight', pad_inches=0.001)

                ## sharpen and gaus blur images/data using convolution on the contrasted images ##
                # plt.figure(1)
                sharpConv = signal.convolve2d(cont_data, sharpen, boundary='fill', mode='same')
                sharpText = filenameShBlr + nseBetaCont + '-Sharp'
                np.save(sharpText, sharpConv)
                # sharp_fig = sns.heatmap(sharpConv, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
                # sharp_img = filenameShBlr + nseBetaCont + '-Sharp.jpg'
                # plt.savefig(sharp_img, bbox_inches='tight', pad_inches=0.001)

                # plt.figure(2)
                blurConv = signal.convolve2d(cont_data, gausBlur, boundary='fill', mode='same')
                blurText = filenameShBlr + nseBetaCont + '-Blur'
                np.save(blurText, blurConv)
                # blur_fig = sns.heatmap(blurConv, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
                # blur_img = filenameShBlr + nseBetaCont + '-Blur.jpg'
                # plt.savefig(blur_img, bbox_inches='tight', pad_inches=0.001)

                # # plt.figure(3)
                # sharpDiff = sns.heatmap((cont_data - sharpConv), vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
                # # plt.figure(4)
                # blurDiff = sns.heatmap((cont_data - blurConv), vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
                # # plt.show()
    execTime = (time.time() - startTime)
    print("Time to run: ", execTime)

totalTime = time.time()-programTime
print("Total Run Time:", totalTime)
