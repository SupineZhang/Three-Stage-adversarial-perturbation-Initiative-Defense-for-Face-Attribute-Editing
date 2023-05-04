import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# from torchsummary import summary
from tqdm import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DenoisingAutoEncoder():
    def __init__(self, image_shape=(3, 128, 128)):
        """
        Denoising autoencoder
        :param image_shape: shape of input image
        """
        self.image_shape = image_shape
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.image_shape[0], out_channels=3, kernel_size=3, padding=1),  # 3*128*128
            nn.Sigmoid(),
            nn.AvgPool2d(2),  # 3*64*64
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),  # 3*64*64
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),  # 3*64*64
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),  # 3*128*128
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),  # 3*128*128
            nn.Sigmoid(),
            nn.Conv2d(in_channels=3, out_channels=self.image_shape[0], kernel_size=3, padding=1),  # 3*128*128
            nn.Sigmoid()
        ).to(device)

    def train(self, data, save_path='./autoencoder/', v_noise=0, num_epochs=100):
        optimizer = torch.optim.Adam(self.model.parameters())
        log_interval = 10
        for epoch in range(num_epochs):
            self.model.train()
            for i, (data_train, data_label) in enumerate(tqdm(data)):
                noise = v_noise * torch.randn_like(data_train)
                noised_data = noise + data_train
                noised_data = torch.clamp(noised_data, min=0, max=1)
                data_train = torch.autograd.Variable(data_train).to(device)
                noised_data = torch.autograd.Variable(noised_data).to(device)
                optimizer.zero_grad()
                output = self.model(noised_data)
                loss = F.mse_loss(output, data_train)
                loss.backward()
                optimizer.step()
                # if i % log_interval == 0:
            print(f'train epoch:{epoch} [{i}/{len(data)} \t loss:{loss.item()}]')
            torch.save(self.model.state_dict(), save_path + 'ae_epoch_' + str(epoch) + '.pth')

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(device)))

    def forward(self, x):
        x_out = self.model(x).to(device)
        return x_out


# if __name__ == '__main__':
#     img = Image.open('./2.jpg')
#     img_tensor = transforms.ToTensor()(img).to(device)
#     img_tensor = img_tensor.unsqueeze(0)
#     image_shape = img_tensor.shape
#     print(image_shape)
#     dae = DenoisingAutoEncoder()
#     dae.load('./ae_epoch_99.pth')
#     img_out = dae.forward(img_tensor)
#     print("img_out_shape:", img_out.shape)
#     img_out = img_out.squeeze(0)
#     print(img_out.shape)
#     img_out = transforms.ToPILImage()(img_out)
#     img_out.save('./ae_2.jpg')







