import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
        os.makedirs(config.sample_dir+'/clean_G')
        os.makedirs(config.sample_dir + '/G')
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.img_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)
    

    # Solver for training and testing
    solver = Solver(celeba_loader, config)

    if config.mode == 'train':
        solver.attack()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #数据集
    parser.add_argument('--celeba_image_dir', type=str, default='./data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--img_size', type=int, default=128, help='image resolution')

    #属性设置
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels')

    #基于attgan的代理模型 SG SD
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')

    #SG SD 训练config
    parser.add_argument('--sm_mode', dest='sm_mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--n_d', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--sm_lr', dest='sm_lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)

    #辅助分类器AC 训练config
    parser.add_argument('--ac_lr', dest='ac_lr', type=float, default=0.0001, help='learning rate')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')


    # Test configuration.
    parser.add_argument('--use_PG', type=str2bool, default=True)
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Attack-Training configurations.
    parser.add_argument('--eps', type=float, default=0.05, help='test model from this step')
    parser.add_argument('--atk_lr', type=float, default=0.0001, help='test model from this step')
    parser.add_argument('--model_iters', type=int, default=140000, help='test model from this step')
    parser.add_argument('--attack_iters', type=int, default=1, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_single_image'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.

    parser.add_argument('--log_dir', type=str, default='attack/logs')
    parser.add_argument('--model_save_dir', type=str, default='attack/models')
    parser.add_argument('--sample_dir', type=str, default='attack/samples')
    parser.add_argument('--result_dir', type=str, default='attack/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)