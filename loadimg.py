import numpy as np
import sys
from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data

def conv_to_single_dim(arr, normalized=True):
    newarr = np.zeros([arr.shape[0], arr.shape[1]])

    newarr = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / 3
    newarr = np.reshape(newarr, [784])
    for i in range(len(newarr)):
        if (normalized):
            newarr[i] = 1 - (newarr[i] / 255)
            newarr[i] = newarr[i] if newarr[i] > 0.5 else 0
        else:
            newarr[i] = (newarr[i] / 255)
    return newarr

def main():
    data = load_image("5.png")

print("calling main")
main()