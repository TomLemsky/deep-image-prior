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

from models import LearnedInput, SmallConvNet
from unet import UNet

def main(args):
    image_fn = "inpainting/hase2_small.jpg" # args.filename
    mask_fn  = "inpainting/hase2_small_mask.jpg"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 3 # 3 Output dimensions (RGB)
    batch_size = 2 # models with batch norm require batch size >=2

    #model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=num_classes)
    #model = models.segmentation.deeplabv3_resnet50(num_classes=num_classes)


    image = Image.open(image_fn)
    target_image = transforms.ToTensor()(image)
    mask = Image.open(mask_fn)
    # just use the first channel of mask image and round to zeroes and ones
    mask_tensor = transforms.ToTensor()(mask)[0].round()
    # zero out masked pixels
    target_image = target_image*mask_tensor

    #img = np.array(image)
    #print(image.size)

    #target_image = transforms.Normalize(0.5,0.5)(target_image)

    # Input pixels are uniformly distributed between 0 and 0.1
    input_image = torch.rand(target_image.size())*0.1
    input_image = input_image.to(device)
    # plt.imshow(target_image.permute((1,2,0)))
    # plt.show()

    target_image = target_image.to(device)
    mask_tensor  = mask_tensor.to(device)


    im_size = target_image.size()


    # input_size = im_size
    # print(input_size)
    # input = torch.rand(input_size)*0.1
    # learned_input = nn.parameter.Parameter(data=input)
    # learned_input = learned_input.to(device)
    # learned_input.requires_grad_(True)
    learned_input = LearnedInput((batch_size,*im_size))
    model = UNet(in_channels=3, num_classes=3,featuresizes=[16,32,64,128,128,128],k_d=3,k_u=5)
    model = nn.Sequential(learned_input,model)
    model = model.to(device)

    params_to_optimize = [{'params': model.parameters()}]
    betas = (args.beta1,args.beta2)
    optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=betas)
    # optimizer = optim.SGD(params_to_optimize, lr=args.lr
    #             , momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)

    criterion = nn.MSELoss()


    #input_batch  = torch.stack(batch_size*[learned_input]) #target_image.view((1,3,128,128))
    target_batch = torch.stack(batch_size*[target_image])
    try:
        for i in range(args.iterations):
            if (i+1)%100==0:
                save_image(output[0,:,:,:],f"outputs/{i:05}.jpg")
                print(i)
            output = model(target_batch)
            # if this is a torchvision model which returns OrderedDict
            if type(output)==dict:
                output = output["out"]
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

        mi = output_image.min()
        ma = output_image.max()
        #plt.imshow((output_image-mi)/(ma-mi))
        #plt.show()
        #save_image((output[0,:,:,:]-mi)/(ma-mi),"inpainted_image3.jpg")
        save_image(output[0,:,:,:],"inpainted_image5.jpg")

def parse_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument('filename', metavar='f', type=str, help='Input image')

    parser.add_argument("--iterations", default=10000, type=int, help="Number of iterations to train for")

    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 value for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 value for Adam')

    parser.add_argument('--architecture', default="deeplabv3", type=str,
                        help='Neural network architecture to use')

    parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                        help='momentum (only when using SGD)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay') # 1e-4

    parser.add_argument('--sched_step', default=999000, type=int,
                    help='after how many iterations to change learning rate')
    parser.add_argument('--sched_gamma', default=0.3, type=float,
                    help='what to multiply learning rate by after sched_step iterations')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
