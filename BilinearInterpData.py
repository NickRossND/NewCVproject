
import numpy as np
import os
from scipy.interpolate import interp2d
import seaborn as sns
import matplotlib.pyplot as plt

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def bilinInterp(normalData):
    x = range(normalData.shape[0])
    y = range(normalData.shape[1])
    print(x)
    print(y)
    ## perform bilinear interpolation of the data ##
    while True:
        try:
            f = interp2d(x, y, original_data, kind="linear", bounds_error=False)
            break
        except ValueError:
            f = interp2d(y, x, original_data, kind="linear", bounds_error=False)
            break    # print(f)
    ## create x,y positions for interpolated data and turn matrix into 6x6 for CNN
    xnew = np.linspace(x[0],x[-1],7, endpoint=False)[1:]
    ynew = np.linspace(y[0],y[-1],7, endpoint=False)[1:]
    print(xnew)
    print(ynew)
    fnew = f(xnew, ynew)
    print(fnew)
    print("original data",original_data)
    print('\n')
    print("interpolated data",fnew)
    return fnew

## define a path where data is stored
path = 'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/DKU02_140508/Right-TOI.txt'
# print(path)
# #extract the last part of the file name without the ext
# #good when visualizing which breast and chromophore you're investigating
# file_name = os.path.splitext(os.path.basename(path))[0]
# print(file_name)
## separate path name from ext ##
my_path, ext = os.path.splitext(path)
# print(my_path)

# # read the data from text file
original_data = np.genfromtxt(path, dtype=float)
# only for "Right" breast files horizontally flip the data so that it matches original data
original_data = np.flip(original_data, 1)

## normalize the data before interpolation ##
norm_ogData = normalize_2d(original_data)
print("norm data", norm_ogData)
newData = bilinInterp(norm_ogData)
# ## use seaborn library for heatmap to confirm that the transformation worked
sns.set()
plt.figure(1)
og = sns.heatmap(original_data, cmap='jet', cbar=False)
plt.title('Original data: ')
plt.figure(2)
plt.title('Interp data: ')
new_img = sns.heatmap(newData, cmap='jet', cbar=False)# , xticklabels=False, yticklabels=False)
plt.show()
