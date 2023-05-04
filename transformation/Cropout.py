import torch
import torch.nn as nn
from .crop import get_random_rectangle_inside
from torchvision import transforms
import cv2


class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # cover_image = noised_and_cover[0]
        # assert noised_image.shape == cover_image.shape

        cropout_mask = torch.ones_like(noised_image)
        # print(cropout_mask.shape)
        # print(cropout_mask)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        # print("h_start:", h_start)
        # print("h_end:", h_end)
        # print("w_start:", w_start)
        # print("w_end:", w_end)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 0
        # print(cropout_mask)

        noised_and_cover[0] = noised_image * cropout_mask
                              # + cover_image * (1-cropout_mask)
        return noised_and_cover


# if __name__ == '__main__':
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     crop_nn = Cropout([0.0, 0.3], [0.0, 0.3])
#     test_sample = cv2.imread('../0.jpg')
#     test_sample = cv2.cvtColor(test_sample, cv2.COLOR_BGR2RGB)
#     # print(test_sample.shape)
#     # print(test_sample)
#     trans = transforms.ToTensor()
#     test_sample = trans(test_sample).to(device)
#     test_sample = torch.unsqueeze(test_sample, 0)
#     # print(test_sample.size())
#     # print(test_sample)
#     test_sample = test_sample * 255.0
#     test_sample = torch.clamp(test_sample, 0, 255)
#     # print([test_sample])
#     # print([test_sample][0].shape)
#     out_sample = crop_nn([test_sample])
#     # print(out_sample)
#     out_sample = out_sample[0] / 255.0
#     # print(out_sample)
#     out_sample = torch.clamp(out_sample, 0, 1)
#     toPIL = transforms.ToPILImage()
#     out_sample = torch.squeeze(out_sample, 0)
#     out_sample = toPIL(out_sample)
#     out_sample.save('../0_crop.jpg')