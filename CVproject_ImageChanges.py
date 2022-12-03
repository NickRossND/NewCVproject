# script to add gaussian noise to an image
import cv2
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# define a path where the image is
path = 'G:/My Drive/Notre Dame/Classes/ComputerVision/CVproject-BreastCancer/SandhyaData/CVProject/PatientPlots/DKU01_140502/Right-TOI.png'
# this is the original image
f = cv2.imread(path)
print(f)
f = f/255
print(f)
cv2.imshow('image', f)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f[1, 1, 1])
#create the noise
dimensions = f.shape

mean = 0
var = 0.1
sigma = np.sqrt(var)
n = np.random.normal(loc=mean,
                     scale=sigma,
                     size=dimensions)
cv2.imshow('Gaussian noise', n)
cv2.waitKey(0)
cv2.destroyAllWindows()

# kde = gaussian_kde(n.reshape(int((-1, -1, 3))))
# dist_space = np.linspace(np.min(n), np.max(n), 100)
# plt.plot(dist_space, kde(dist_space))
# plt.xlabel('Noise pixel value'); plt.ylabel('Frequency')
# plt.show()
img = f+n
cv2.imshow('image with noise', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# change the brightness and contrast of the image
# alpha is contrast and beta is brightness in file(x) = ax + b
alpha = 1.0
beta = 0.5

new_image = cv2.addWeighted(f,alpha,np.zeros(f.shape, f.dtype),0,beta)

cv2.imshow("new", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
