
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
#import helper
import time
from PIL import Image
import seaborn as sb



def show_images(loader):
    images, labels = next(iter(loader))
    print(images.shape)
    print(labels)
    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        helper.imshow(images[ii], ax=ax)

def load_cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
  
    # resize (keep ratio to 256)
    factor = 256.00 / min(image.width, image.height)
    size = (int(image.width * factor), int(image.height * factor))
    #print("size = ", size)
    image = image.resize(size)
    #print("image size: ", image.size)
    # central crop
    padding_width = (image.width - 224) / 2
    padding_height = (image.height - 224) / 2
    box = (padding_width, padding_height, padding_width + 224, padding_height + 224)
    #print("box = ", box)
    image = image.crop(box)   
    #print("image size: ", image.size)
    # 0-255 map to 0-1
    np_image = np.array(image) / 255
    # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std      
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax