import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageFilter


def ImgLoader_rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def ImgLoader_28X28InvGray(path):
    img = Image.open(path).filter(ImageFilter.GaussianBlur(1.1))
    img=img.resize((28,28))
    img = np.asarray(img)
    gray = ImgLoader_rgb2gray(img)
    gray = 1-gray.flatten()/255
    #plt.imshow(gray, cmap = plt.get_cmap('gray'))
    #plt.show()
    return gray
