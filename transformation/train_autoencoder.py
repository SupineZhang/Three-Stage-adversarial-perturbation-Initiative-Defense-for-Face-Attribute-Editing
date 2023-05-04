from AutoEncoder import DenoisingAutoEncoder as DAE
from data_preprocessing import data_preprocess
from utils import read_data


train_image_path, train_image_label = read_data('../StarGAN/results/train/fake_stargan')
train_data_set, train_loader = data_preprocess(train_image_path, train_image_label, 128, 64)

dae = DAE()
dae.train(train_loader, v_noise=0.1, num_epochs=200)

print("training has completed.")