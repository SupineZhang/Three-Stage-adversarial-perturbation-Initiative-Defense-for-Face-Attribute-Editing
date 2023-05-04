from PIL.Image import Image
import torch.nn as nn
import torch
import random
from torchvision import transforms
import cv2

# define
sizepic = [0, 0]
timer = [0, 0, 0, 0]
PI = 3.1415926

from PIL import Image
import numpy as np
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kornia.filters import GaussianBlur2d

# from time import clock


class Gaussian(nn.Module):
    def __init__(self,sigma=0.8,kernel=3):
       super(Gaussian, self).__init__()
       self.gussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

    def forward(self, input):
        noise = self.gussian_filter(input)

        return noise


# if __name__ == "__main__":
#     img = cv2.imread('../0_adv.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = transforms.ToTensor()(img)
#     img = torch.unsqueeze(img, 0)
#     print(img.shape)
#
#     gau_blur = Gaussian()
#     out = gau_blur(img)
#     out = torch.squeeze(out, 0)
#     out = transforms.ToPILImage()(out)
#     out.save('../0_adv_blur.png')

