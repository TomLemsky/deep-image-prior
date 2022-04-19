import torch
import torch.nn as nn
import torch.nn.functional as func

def get_convblock(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batchnorm=True):
    activation_function = nn.LeakyReLU
    if padding=='same':
        padding = (kernel_size-1)//2
    conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding, stride=stride, bias=True)
    if batchnorm:
        layers = [conv, nn.BatchNorm2d(out_channels), activation_function(inplace=True)]
    else:
        layers = [conv, activation_function(inplace=True)]
    return nn.Sequential(*layers)

def make_up_conv(in_channels,out_channels, kernel_size=2): #, padding=1):
    upscale = nn.Upsample(scale_factor=2, mode='nearest') #bilinear
    zeropad = nn.ZeroPad2d((0,1,0,1)) # pad only right and bottom to keep dimension after 2x2 conv
    conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=0)
    return nn.Sequential(upscale, zeropad, conv)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, init_weights=True,featuresizes=None,k_d=3,k_u=3):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        unet_in_channels = in_channels

        if featuresizes is None:
            featuresizes = [64,128,256,512,1024]

        # with normal Python lists, weights wouldn't be registered
        self.contracting_path = nn.ModuleList([])
        self.expansive_path   = nn.ModuleList([])
        self.up_convs         = nn.ModuleList([])

        # contracting_path
        for i in range(len(featuresizes)):

            current = featuresizes[i]
            # the first feature dimension is the input channel dimension
            if i>0:
                previous = featuresizes[i-1]
            else:
                previous = in_channels
            #print(previous, current)
            conv_1 = get_convblock(previous,current,kernel_size=k_d,padding='same')
            conv_2 = get_convblock(current ,current,kernel_size=k_d,padding='same')
            combined_block = nn.Sequential(conv_1,conv_2)
            self.contracting_path.append(combined_block)

        #print("!")

        # same loop in reverse (without input channels)
        for i in range(1,len(featuresizes))[::-1]:

            # previous is now the larger filter size, since we are upscaling the image on the expansive_path
            previous = featuresizes[i]
            # because the expansive_path goes reverse to the contracting_path
            current  = featuresizes[i-1]
            # contracting_path
            upconv = make_up_conv(previous,current)
            self.up_convs.append(upconv)
            block_inputsize = 2*current
            conv_1 = get_convblock(block_inputsize,current,kernel_size=k_u,padding='same')
            conv_2 = get_convblock(current, current,kernel_size=k_u,padding='same')
            combined_block = nn.Sequential(conv_1,conv_2)
            self.expansive_path.append(combined_block)

        zeropad = nn.ZeroPad2d((0,1,0,1)) # pad only right and bottom to keep dimension after 2x2 conv
        final_conv = nn.Conv2d(featuresizes[0],num_classes,kernel_size=2)
        self.classifier = nn.Sequential(zeropad,final_conv)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        unet_input = x
        _, _, in_height, in_width = x.shape

        contracting_outputs = []
        n = len(self.contracting_path)
        for i, block in enumerate(self.contracting_path):
            x = block(x)
            if i < n-1: # last level doesn't have pooling or skip connections
                contracting_outputs.append(x)
                x = func.max_pool2d(x,kernel_size=2, stride=2, padding=0, ceil_mode=True)

        for upconv, skip, block in zip(self.up_convs,contracting_outputs[::-1],self.expansive_path):
            x = upconv(x)
            # up-conved features together with skip-connection
            x = torch.hstack([x,skip])
            x = block(x)

        features = func.interpolate(x, size=(in_height,in_width), mode='bilinear')
        return self.classifier(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
