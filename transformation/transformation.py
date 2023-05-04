import random
from transformation.JPEG.DiffJPEG import DiffJPEG
from .GaussianBlur import Gaussian
from .Cropout import Cropout
from .AutoEncoder import DenoisingAutoEncoder
from torchvision.transforms import Resize
from .GaussNoise import Gaussnoise
import torch
import torch.nn as nn

def selectAttack(adv, number, device='cpu'):
    if number == 0:
        return adv
    if number == 1:
        jpeg_nn = DiffJPEG(height=128, width=128, differentiable=True, quality=80).to(device)
        noise = jpeg_nn(adv)
        return noise
    elif number == 2:
        gussblur_nn = Gaussian()
        noise = gussblur_nn(adv)
        return noise
    elif number == 6:
        cropout_nn = Cropout([0.0, 0.3], [0.0, 0.3])
        adv = adv * 255.0
        adv = torch.clamp(adv, 0, 255)
        noise = cropout_nn([adv])
        noise = noise[0] / 255.0
        return noise
    elif number == 4:
        dae_nn = DenoisingAutoEncoder()
        dae_nn.load('/data1/lyr/one/transformation/autoencoder/ae_epoch_199.pth')
        noise = dae_nn.forward(adv)
        return noise
    elif number == 5:
        rnd = random.randint(100, 128)
        resize_nn = Resize([rnd, rnd])
        pad_width = 128 - rnd
        pad_height = 128 - rnd
        w1 = random.randint(0, pad_width)
        w2 = pad_width - w1
        h1 = random.randint(0, pad_height)
        h2 = pad_height - h1
        resized_img = resize_nn(adv)
        pad = nn.ZeroPad2d(padding=(w1, w2, h1, h2))
        noise = pad(resized_img)
        return noise
    elif number == 3:
        gussnoise_nn = Gaussnoise()
        adv = adv * 255.0
        adv = torch.clamp(adv, 0, 255)
        noise = gussnoise_nn([adv])
        noise = noise[0] / 255.0
        return noise








