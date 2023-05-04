# from .Resize import Resize
# from .resize_dim import Resize
import random
from torchvision.transforms import Resize
import torch.nn as nn
# from .gaussian_blur import Gaussian
import cv2
from torchvision import transforms
import torch
import numpy as np


def input_diversity(input):
    # resize padding
    # resize_nn = Resize([0.7, 0.8])
    # input = input * 255.0
    # resized_img = resize_nn([input])
    # resized_img = resized_img[0] / 255.0

    # scale
    # m = random.randint(1, 5)
    # resized_img = input / 2 ** m

    # resize padding
    rnd = random.randint(100, 128)
    resize_nn = Resize([rnd, rnd])
    pad_width = 128 - rnd
    pad_height = 128 -rnd
    w1 = random.randint(0, pad_width)
    w2 = pad_width - w1
    h1 = random.randint(0, pad_height)
    h2 = pad_height - h1
    resized_img = resize_nn(input)
    pad = nn.ZeroPad2d(padding=(w1, w2, h1, h2))
    resized_img = pad(resized_img)

    # gaussian noise
    # input = input.detach().cpu().numpy()
    # noise = np.random.normal(loc=0.0, scale=0.02, size=(input.shape))
    # resized_img = input + noise
    # resized_img = torch.from_numpy(resized_img)
    # resized_img = resized_img.float()

    # gaussian blur
    # gau_blur = Gaussian()
    # resized_img = gau_blur(input)

    return resized_img


# if __name__ == '__main__':
#     img = cv2.imread('../0_adv.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = transforms.ToTensor()(img)
#     img = torch.unsqueeze(img, 0)
#
#     # img = img.detach().cpu().numpy()  # 张量转图片
#     print(img.shape)
    # resized_img = input_diversity(img)
#     # resized_img = torch.from_numpy(resized_img)  # 图片转张量
#     resized_img = torch.clamp(resized_img, 0, 1)
#     resized_img = torch.squeeze(resized_img, 0)
#     resized_img = transforms.ToPILImage()(resized_img)
#     resized_img.save('../0_adv_gn.png')

