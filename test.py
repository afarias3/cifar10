import argparse
import os, datetime
import torch

import numpy as np

from dataio import NoisyCIFAR10Dataset
from torch.utils.data import DataLoader
from denoising_unet import DenoisingUnet
from tensorboardX import SummaryWriter

# params
parser = argparse.ArgumentParser()

# data paths
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')

# train params
parser.add_argument('--train_test', type=str, required=False, help='path to file list of h5 train data')
parser.add_argument('--experiment_name', type=str, default='', help='path to file list of h5 train data')
parser.add_argument('--checkpoint', type=str, default=None, help='path to file list of h5 train data')
parser.add_argument('--max_epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--sigma', type=float, default=0.05, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=4, help='start epoch')

parser.add_argument('--reg_weight', type=int, default=0., help='start epoch')

opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, dataset):
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint))

    model.eval()
    model.to(device)

    print('Beginning testing...')
    total_loss_arr = []
    for model_input, ground_truth in dataloader:
        ground_truth = ground_truth.to(device)
        model_input = model_input.to(device)

        model_outputs = model(model_input)

        dist_loss = model.get_distortion_loss(model_outputs, ground_truth)
        reg_loss = model.get_regularization_loss(model_outputs, ground_truth)

        total_loss = dist_loss + opt.reg_weight * reg_loss
        total_loss_arr.append(torch.mean(total_loss).item())
        print(f'instantaneous testing average loss.{np.mean(total_loss_arr).item():.10f}')
    print(f'testing average loss.{np.mean(total_loss_arr).item():.10f}')

def main():
    #dataset = NoisyCIFAR10Dataset(data_root=opt.data_root,
    #                             sigma=opt.sigma,
    #                             train=opt.train_test == 'train')
    dataset = NoisyCIFAR10Dataset(data_root=opt.data_root,
                                 sigma=opt.sigma,
                                 train=False)
    model = DenoisingUnet(img_sidelength=32)
    test(model, dataset)


if __name__ == '__main__':
    main()

