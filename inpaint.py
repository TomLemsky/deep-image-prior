import argparse
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim

from unet import UNet

class LearnedInput(nn.Module):
    """PyTorch module that just outputs its learned Parameters of the specified dimensions"""
    def __init__(self, dimensions):
        super(LearnedInput, self).__init__()
        input = torch.rand(dimensions)*0.1
        self.learned_input = nn.parameter.Parameter(data=input)
    def forward(self,x):
        return self.learned_input

def main(args):
    # Use GPU if available, otherwise cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 3 # Three output dimensions (RGB)
    batch_size = 2 # models with batch norm require batch size >=2

    # Load image and mask
    image = Image.open(args.image_file)
    target_image = transforms.ToTensor()(image)
    mask = Image.open(args.mask_file)
    # just use the first channel of mask image and round to zeroes and ones
    mask_tensor = transforms.ToTensor()(mask)[0].round()
    # zero out masked pixels
    target_image = target_image*mask_tensor

    # Input pixels are uniformly distributed between 0 and 0.1
    input_image = torch.rand(target_image.size())*0.1
    input_image = input_image.to(device)

    target_image = target_image.to(device)
    mask_tensor  = mask_tensor.to(device)

    im_size = target_image.size()

    learned_input = LearnedInput((batch_size,*im_size))
    model = UNet(in_channels=3, num_classes=3,featuresizes=[16,32,64,128,128,128],k_d=3,k_u=5)
    model = nn.Sequential(learned_input,model)
    model = model.to(device)

    params_to_optimize = [{'params': model.parameters()}]
    betas = (args.beta1,args.beta2)
    optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=betas)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

    criterion = nn.MSELoss()

    target_batch = torch.stack(batch_size*[target_image])

    # try block allows user to interrupt training with Ctrl+C, but still get an output
    try:
        for i in range(args.iterations):
            if (i+1)%200==0:
                print(f'Iteration {i}')
            # Store every a-th iteration as an image for animation
            # Default value a=-1 should never save a frame
            if (i%args.animation_frames)-1 == 0:
                save_image(output[0,:,:,:],f'outputs/{i:05}.jpg')

            output = model(target_batch)
            # remove masked pixels from output so that they don't affect loss
            masked_output = output*mask_tensor
            loss = criterion(masked_output,target_batch)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    finally:
        output_image = output[0,:,:,:].permute((1,2,0)).detach().cpu()

        save_image(output[0,:,:,:],args.output)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_file', metavar='i', type=str, help='Input image path')
    parser.add_argument('mask_file' , metavar='m', type=str, help='Input mask path')

    parser.add_argument('-o', '--output', default='inpainted_image.jpg', type=str, help='Output path')

    parser.add_argument('--animation_frames', default=-1, type=int, help='Store every a-th iteration as an image for animation')

    parser.add_argument("--iterations", default=10000, type=int, help="Number of iterations to train for")

    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 value for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 value for Adam')

    parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                        help='momentum (only when using SGD)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay') # 1e-4

    # Scheduling is not actually useful for Deep Image Prior
    parser.add_argument('--sched_step', default=999999999, type=int,
                    help='after how many iterations to change learning rate')
    parser.add_argument('--sched_gamma', default=0.3, type=float,
                    help='what to multiply learning rate by after sched_step iterations')

    args = parser.parse_args()


    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
