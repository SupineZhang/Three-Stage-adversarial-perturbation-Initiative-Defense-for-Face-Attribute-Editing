import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import transforms


def gaussnoise(image, scale):

    noise = np.random.normal(loc=0.0, scale=scale, size=(128, 128))  # mean 0.0, std 1
    decimg = np.clip(np.asarray(image, float) + np.tile(np.expand_dims(noise, 0), (3, 1, 1)), 0.0, 255.0)

    return decimg


class Gaussnoise(nn.Module):

    def __init__(self):
        super(Gaussnoise, self).__init__()

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        batch, channel, col, row = noised_image.shape
        scale = 5
        for i in range(batch):
            noised_image_np = noised_image[i].detach().cpu().numpy()  # 张量转图片
            noised_image_np = gaussnoise(noised_image_np, scale)  # 图片做高斯
            noised_image_np = torch.from_numpy(noised_image_np)  # 图片转张量
            noised_image[i] = noised_image_np
            noised_and_cover[0] = noised_image

        ## 不要硬编码

        return noised_and_cover


# if __name__ == '__main__':
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     gussnoise_nn = Gaussnoise()
#     test_sample = cv2.imread('../1425.jpg')
#     test_sample = cv2.cvtColor(test_sample, cv2.COLOR_BGR2RGB)
#     trans = transforms.ToTensor()
#     test_sample = trans(test_sample).to(device)
#     test_sample = torch.unsqueeze(test_sample, 0)
#     test_sample = test_sample * 255.0
#     test_sample = torch.clamp(test_sample, 0, 255)
#     out_sample = gussnoise_nn([test_sample])
#     out_sample = out_sample[0] / 255.0
#     out_sample = torch.clamp(out_sample, 0, 1)
#     # print(out_sample.shape)
#     toPIL = transforms.ToPILImage()
#     out_sample = torch.squeeze(out_sample, 0)
#     out_sample = toPIL(out_sample)
#     out_sample.save('../1425_noise25.jpg')


