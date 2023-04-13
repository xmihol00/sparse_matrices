import numpy as np
from scipy import ndimage
import idx2numpy as idx

scale_factor = 32 / 28

image_array = idx.convert_from_file("../mnist_ML/mnist/train-images.idx3-ubyte") / 255
upscaled_array = np.zeros((image_array.shape[0], 32, 32))

for i, img in enumerate(image_array):
    upscaled_img = ndimage.zoom(img, scale_factor, order=1)
    upscaled_array[i] = upscaled_img

np.save("datasets/upscaled_mnist_train.npy", upscaled_array)

image_array = idx.convert_from_file("../mnist_ML/mnist/t10k-images.idx3-ubyte") / 255
upscaled_array = np.zeros((image_array.shape[0], 32, 32))

for i, img in enumerate(image_array):
    upscaled_img = ndimage.zoom(img, scale_factor, order=1)
    upscaled_array[i] = upscaled_img

np.save("datasets/upscaled_mnist_test.npy", upscaled_array)
