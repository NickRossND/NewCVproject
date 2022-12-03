# script to add gaussian noise to an data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import seaborn as sns
import os
import glob

# define a path where the image is
path = 'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/DKU01_140502/Right-TOI.txt'
# print(path)
file_name = os.path.splitext(os.path.basename(path))[0]
print(file_name)
my_path, ext = os.path.splitext(path)
print(my_path)

# # read the data from text file
# original_data = np.genfromtxt(path, dtype=float)
# # horizontally flip the data so that it matches original data
# original_data = np.flip(original_data, 1)
# print(original_data)
# print('\n')
# #
# # newfile_object = np.divide(original_data, 255)
# # print(newfile_object)
#
# # #create the noise
# dimensions = original_data.shape
# # print("Matrix dimension: ", dimensions)
# # print("\n")
#
# #set up gaussian noise to be added to the data
# mean = 0
# var = 0.1
# sigma = np.multiply(original_data, var)
# # print(sigma)
# noise = np.random.normal(loc=mean,
#                          scale=sigma,
#                          size=dimensions)
# # print("This is noise:\n", noise)
# # print("\n")
#
# # create new matrix with noise added to original data
# noisy_data = original_data + noise
# # print("This is noisy data:\n", noisy_data)
#
# # use seaborn library for heatmap to confirm that the transformation worked
# file = 'C:/Users/Nick/PycharmProjects/pythonProject/Noise10-'+file_name+'.png'
# sns.set()
# plt.figure(1)
# og = sns.heatmap(original_data, cmap='jet')
# plt.figure(2)
# noisy_img = sns.heatmap(noisy_data, cmap='jet')
# plt.savefig(file)
# plt.show()
# #
# #
# # # def addBrightness (image_name)
# # # change the brightness and contrast of the image
# # # alpha is contrast and beta is brightness in file(x) = ax + b
# # # alpha = 1.0
# # # beta = 0.5
# # # new_image = cv2.addWeighted(axs1, alpha, np.zeros(axs1.shape, axs1.dtype), 0, beta)
# # # #new_image = cv2.addWeighted(image_name, alpha, np.zeros(image_name.shape, image_name.dtype), 0, beta)
# # # print("\n", new_image)
# # # cv2.imshow("new", new_image)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
# #
