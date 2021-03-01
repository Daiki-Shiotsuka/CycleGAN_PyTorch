import argparse
import os
import sys
import itertools
import math
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from PIL import Image

from model import CycleGenerator, Discriminator
from lr_helpers import get_lambda_rule
from dataset import CycleGANDataset
from train import create_parser

def test_loop(opts):

    if opts.image_height == 128:
        res_blocks = 6
    elif opts.image_height >= 256:
        res_blocks = 9

    transform = transforms.Compose([transforms.Resize(int(opts.image_height*1.12), Image.BICUBIC),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    # Create networks
    G_AB = CycleGenerator(opts.a_channels, opts.b_channels, res_blocks).to(device)
    G_BA = CycleGenerator(opts.b_channels, opts.a_channels, res_blocks).to(device)
    D_A = Discriminator(opts.a_channels, opts.d_conv_dim).to(device)
    D_B = Discriminator(opts.b_channels, opts.d_conv_dim).to(device)

    G_AB.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_G_AB.pth"))
    G_BA.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_G_BA.pth"))

    test_dataloader = DataLoader(CycleGANDataset(opts.dataset_name, transform, mode='test'), batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        if not os.path.exists("result/" + opts.dataset_name):
            os.makedirs("result/" + opts.dataset_name)
        for index, batch in enumerate(test_dataloader):
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            fake_B2A = G_BA(real_B)
            fake_A2B = G_AB(real_A)

            rec_A, rec_B = G_BA(fake_A2B), G_AB(fake_B2A)

            save_image(real_A, f"result/{opts.dataset_name}/{str(index)}_A_real.png",normalize=True)
            save_image(real_B, f"result/{opts.dataset_name}/{str(index)}_B_real.png",normalize=True)
            save_image(fake_A2B, f"result/{opts.dataset_name}/{str(index)}_A2B_fake.png",normalize=True)
            save_image(fake_B2A, f"result/{opts.dataset_name}/{str(index)}_B2A_fake.png",normalize=True)
            save_image(rec_A, f"result/{opts.dataset_name}/{str(index)}_A_rec.png",normalize=True)
            save_image(rec_B, f"result/{opts.dataset_name}/{str(index)}_B_rec.png",normalize=True)
            print('calculate...'+str(index))


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')

    test_loop(opts)
