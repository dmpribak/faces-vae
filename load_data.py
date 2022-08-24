from PIL import Image
import numpy as np
import torch

path = "X:/Datasets/img_align_celeba/img_align_celeba/"

def load_img(start, end):
    pass

def load_data(size, chunk):
    images = []
    for i in range(1+chunk*size,(chunk+1)*size+1):
        num = (6-len(str(i)))*"0" + str(i)
        image = Image.open(path + num + ".jpg")
        images.append(np.array(image))
    return torch.tensor(np.array(images), dtype=torch.float32).permute(0, 3, 1, 2).cuda()