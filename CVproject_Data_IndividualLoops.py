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
            if os.path.isdir(newpath) == False and file_name.startswith("DK1"):
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
# define a path where the image is
dirpath = r'C:\Users\Nick\Downloads\PatientPlots\**'
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
startTime = time.time()
#
# for file in txtFiles:
#     ## Separate out different parts of the path name to create new file names later
#     # file_name = os.path.splitext(os.path.basename(file))[0]
#     startTime = time.time()
#     print("curent folder", file)
#     dirName =os.path.dirname(file)
#     file_name = os.path.splitext(os.path.basename(file))[0]
#     my_path, ext = os.path.splitext(file)
#     # print(file_name)
#
#     ### used to create new folders in each patient folder ###
#     interpFldr = '\\Interpolated\\'
#     filenameInterp = dirName + interpFldr + file_name
#
#     # read the data from text file
#     original_data = np.genfromtxt(file, dtype=float)
#     # only for "Right" breast files
#     if fnmatch.fnmatch(file, '*Right*'):
#         # print(file)
#         ## horizontally flip the data so that it matches structure of original data
#         original_data = np.flip(original_data, 1)
#
#     # print("Matrix dimension: ", dimensions)
#     ## normalize the data before interpolation ##
#     norm_ogData = normalize_2d(original_data)
#     interpData = bilinInterp(norm_ogData)
#     dimensions = interpData.shape
#     np.savetxt(filenameInterp+'-INTERP.txt', interpData, fmt = '%1.9f')
#     interp_fig = sns.heatmap(interpData, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
#     interp_img = filenameInterp + '-INTERP.jpg'
#     plt.savefig(interp_img, bbox_inches='tight', pad_inches=0.001)
#     # print("Data normalized and interpolated")
#
# interpPath = r'C:\Users\Nick\Downloads\PatientPlots\**\Interpolated\*.txt'
# interpFiles = glob.glob(interpPath, recursive=False)
# print(interpFiles)

# for terp in interpFiles:
#     interpData = np.genfromtxt(terp)
#     # print("File: ", terp)
#     # print('\n', interpData)
#     dirName =os.path.dirname(os.path.dirname(terp))
#     # print(dirName)
#     file_name = os.path.splitext(os.path.basename(terp))[0]
#     file_name = file_name[:-6]
#     # print(file_name)
#     nseFldr = '\\Noise\\'
#     filenameNoise = dirName + nseFldr + file_name
#     # print(filenameNoise)
#     ## create the noise ##
#     mean = 0
#     for x in percentAmt:
#         ## set up gaussian noise to be added to the data
#         var = 0.01*x
#         sigma = np.multiply(interpData, var)
#         #print(sigma)
#         noise = np.random.normal(loc=mean,
#                                  scale=sigma,
#                                  size=dimensions)
#         # print("This is noise:\n", noise)
#
#         ## create new matrix with noise added to original data
#         noisy_data = interpData + noise
#         noisy_txt = filenameNoise + 'NOISE' + (str(int(var * 100))) + '.txt'
#         # print("Noise:\n", noisy_data)
#         ### Check if there are already files with noise in the name and create new files/imgs if they don't exist ##
#         # if "NOISE" or "BRT" or "CONTRAST" or "Sharp" or "Blur" not in file_name:
#         np.savetxt(noisy_txt, noisy_data, fmt = '%1.9f')
#         #     # ## use seaborn library for heatmap to confirm that the transformation worked
#         #     sns.set()
#         #     plt.figure(1)
#         #     og = sns.heatmap(original_data, cmap='jet', cbar=False)
#         #     plt.figure(2)
#         #     # plt.title('Noisy data: ' + file_name)
#         noisy_fig = sns.heatmap(noisy_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
#         nsy_img = filenameNoise + 'NOISE'+(str(int(var*100)))+'.jpg'
#         plt.savefig(nsy_img, bbox_inches='tight', pad_inches=0.001)
#         #     print("new file: ", noisy_txt)
#         newNoisy = torch.tensor(noisy_data)
#         #
## run through each noise file and add brightness to each image ##
nsyPath = r'C:\Users\Nick\Downloads\PatientPlots\**\Noise\*.txt'
nsyFiles = glob.glob(nsyPath, recursive=False)
# print(nsyFiles)
# for nsy in nsyFiles:
#     noisy_data = np.genfromtxt(nsy)
#     # print("File: ", nsy)
#     # print('\n', noisy_data)
#     dirName =os.path.dirname(os.path.dirname(nsy))
#     # print(dirName)
#     file_name = os.path.splitext(os.path.basename(nsy))[0]
#     # print(file_name)
#     brtFldr = '\\Brightness\\'
#     filenameBrt = dirName + brtFldr + file_name + '-BRT'
#     # print(filenameBrt)
#     for y in beta:
#         temp = torch.tensor(noisy_data + y)
#         temp2 = torch.ones(noisy_data.shape)
#         brt_data = torch.where(temp > 1, temp2, temp)
#         # print("with beta\n", beta_data )
#         brt_txt = filenameBrt + str(int(y * 100)) + '.txt'
#         # print(brt_txt)
#         np.savetxt(brt_txt, brt_data, fmt ='%1.9f')
#         brt_fig = sns.heatmap(brt_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
#         brt_img = filenameBrt + str(int(y * 100)) +'.jpg'
#         plt.savefig(brt_img, bbox_inches='tight', pad_inches=0.001)

## run through each noise file and add brightness to each image ##
brtPath = r'C:\Users\Nick\Downloads\PatientPlots\**\Brightness\*.txt'
brtFiles = glob.glob(brtPath, recursive=False)
# print(nsyFiles)
# for brt in brtFiles:
#     brt_data = np.genfromtxt(brt)
#     # print("File: ", nsy)
#     # print('\n', noisy_data)
#     dirName =os.path.dirname(os.path.dirname(brt))
#     # print(dirName)
#     file_name = os.path.splitext(os.path.basename(brt))[0]
#     # print(file_name)
#     contFldr = '\\Contrast\\'
#     filenameCont = dirName + contFldr + file_name + '-CONT'
#     print(filenameCont)
# # ## Add contrast to each image that was just brightened ##
#     for z in alpha:
#         temp = torch.tensor(brt_data * z)
#         temp2 = torch.ones(brt_data.shape)
#         cont_data = torch.where(temp > 1, temp2, temp)
#         # print(cont_data)
#         cont_txt = filenameCont + str(int(z * 100)) + '.txt'
#         np.savetxt(cont_txt, cont_data.numpy(), fmt ='%1.9f')
#         cont_img = filenameCont + str(int(z * 100)) + '.jpg'
#         cont_fig = sns.heatmap(cont_data, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
#         plt.savefig(cont_img, bbox_inches='tight', pad_inches=0.001)


contPath = r'C:\Users\Nick\Downloads\PatientPlots\**\Contrast\*.txt'
contFiles = glob.glob(contPath, recursive=False)
# print(contFiles)
for cont in contFiles:
    cont_data = np.genfromtxt(cont)
    # print("File: ", nsy)
    # print('\n', cont_data)
    dirName =os.path.dirname(os.path.dirname(cont))
    # print(dirName)
    file_name = os.path.splitext(os.path.basename(cont))[0]
    # print(file_name)
    sharpBlurFldr = '\\Sharp_Blur\\'
    filenameShBlr = dirName + sharpBlurFldr + file_name
    print(filenameShBlr)
    ## sharpen and gaus blur images/data using convolution on the contrasted images ##
    # plt.figure(1)
    sharpConv = signal.convolve2d(cont_data, sharpen, boundary='symm', mode='same')
    sharpText = filenameShBlr + '-Sharp.txt'
    np.savetxt(sharpText, sharpConv, fmt='%1.9f')
    sharp_fig = sns.heatmap(sharpConv, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    sharp_img = filenameShBlr + '-Sharp.jpg'
    plt.savefig(sharp_img, bbox_inches='tight', pad_inches=0.001)

    # plt.figure(2)
    blurConv = signal.convolve2d(cont_data, gausBlur, boundary='symm', mode='same')
    blurText = filenameShBlr + '-Blur.txt'
    np.savetxt(blurText, blurConv, fmt='%1.9f')
    blur_fig = sns.heatmap(blurConv, vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    blur_img = filenameShBlr + '-Blur.jpg'
    plt.savefig(blur_img, bbox_inches='tight', pad_inches=0.001)
    #
    # plt.figure(3)
    # sharpDiff = sns.heatmap((cont_data - sharpConv), vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    # plt.figure(4)
    # blurDiff = sns.heatmap((cont_data - blurConv), vmin=0, vmax=1, cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
    # plt.show()
execTime = (time.time() - startTime)
print("Time to run: ", execTime)
# noisy_img = cv2.imwrite('C:/Users/Nick/PycharmProjects/pythonProject/Noise10-'+file_name+'.png', noisy_img)

# def addBrightness (image_name)
# change the brightness and contrast of the image
# alpha is contrast and beta is brightness in file(x) = ax + b
# alpha = 1.0
# beta = 0.5
# new_image = cv2.addWeighted(axs1, alpha, np.zeros(axs1.shape, axs1.dtype), 0, beta)
# #new_image = cv2.addWeighted(image_name, alpha, np.zeros(image_name.shape, image_name.dtype), 0, beta)
# print("\n", new_image)
# cv2.imshow("new", new_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# applying gaus blur or sharpen and then subtract the image from the results to see if the image is being destroyed or sign altered
