'''
Micro Object Detector Net
the author:Luis
date : 11.11
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from .base_models import vgg, vgg_base


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.in_channels = in_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class MOD(nn.Module):
    def __init__(self, base, extras, upper,head,num_classes,size):
        super(MOD, self).__init__()
        self.num_classes = num_classes
        self.extras = nn.ModuleList(extras)
        self.size = size
        self.base = nn.ModuleList(base)
        #self.L2Norm = nn.ModuleList(extras)
        self.upper = nn.ModuleList(upper)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax()

    def forward(self, x,test=False):
        
        scale_source = []
        upper_source = []
        loc = []
        conf = []
        mid_trans = []
        #get the F.T of conv4
        for k in range(23):
            x = self.base[k](x)
        scale_source.append(x)
        for k in range(23,len(self.base)):
            x = self.base[k](x)
        scale_source.append(x)
        for k,v in enumerate(self.extras):
            x = F.relu(v(x),inplace=True)
            if k%2 == 1:
                scale_source.append(x)
        upper_source = scale_source
        lenscale = len(scale_source)
        orgin = x
        for k in range(len(self.upper)-1):
            #bn = nn.BatchNorm2d(self.upper[lenscale-k-2].in_channels,affine=True)
            #print(self.upper[lenscale-k-2].in_channels)
            #print(self.upper[lenscale-k-1].out_channels)
            #print(scale_source[lenscale-k-2].size())
            upper_source[0] =upper_source[0]+  self.upper[lenscale-k-1](upper_source[lenscale-k-1])
        bn = nn.BatchNorm2d(512,affine = True)
        upper_source[0] = bn(upper_source[0])
        for (x, l, c) in zip(upper_source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #print(loc.size())
        #print(conf.size())
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
            
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            #print(loc.size())
            #print(conf.size())
        return output

def low_pooling(vgg, extracts, size):
    if size == 300:
        up_size = layer_size('300')[k]
    elif size ==512:
        up_size = layer_size('512')[k]
    layers = []
    
    
def upper_deconv(vgg, extracts, size):    
    layers = []
    if size == 300:
        layers.append(BasicConv(512, 128*4, kernel_size=1, padding=0))
        layers+=[(BasicConv(vgg[-2].out_channels,512,kernel_size=1,padding=0,up_size = 38))]
        layers.append(BasicConv(extracts[1].out_channels,512,kernel_size=1,padding=0,up_size = 38))
        layers.append(BasicConv(extracts[3].out_channels,512,kernel_size=1,padding=0,up_size = 38))
        layers.append(BasicConv(extracts[5].out_channels,512,kernel_size=1,padding=0,up_size = 38))
        layers.append(BasicConv(extracts[7].out_channels,512,kernel_size=1,padding=0,up_size = 38))
    elif size ==512:
        layers.append(BasicConv(512, 128*4, kernel_size=1, padding=0))
        layers.append(BasicConv(vgg[-2].out_channels,512,kernel_size=1,padding=0,up_size = 64))
        layers.append(BasicConv(extracts[1].out_channels,512,kernel_size=1,padding=0,up_size = 64))
        layers.append(BasicConv(extracts[3].out_channels,512,kernel_size=1,padding=0,up_size = 64))
        layers.append(BasicConv(extracts[5].out_channels,512,kernel_size=1,padding=0,up_size = 64))
        layers.append(BasicConv(extracts[7].out_channels,512,kernel_size=1,padding=0,up_size = 64))
        layers.append(BasicConv(extracts[9].out_channels,512,kernel_size=1,padding=0,up_size = 64))
    return vgg, extracts,layers


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    #print(len(layers))
    return layers

def multibox(vgg,extra_layers, upper,cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24,-2]
    loc_layers += [nn.Conv2d(upper[0].out_channels,cfg[0] * 4,kernel_size=3,padding=1)]
    conf_layers += [nn.Conv2d(upper[0].out_channels,cfg[0]*num_classes,kernel_size=3,padding=1)]

    for k,v in enumerate(upper):
        if k ==0:
          continue
        loc_layers += [nn.Conv2d(v.in_channels,cfg[k] * 4,kernel_size=3,padding=1)]
        conf_layers += [nn.Conv2d(v.in_channels,cfg[k]*num_classes,kernel_size=3,padding=1)]
    '''
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    '''
    return vgg, extra_layers, upper, (loc_layers, conf_layers)

layer_size = {
        '300':[38,19,10,5,3,1],
        '512':[64,32,16,8,4,2,1],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}
def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return MOD(*multibox(*upper_deconv(vgg(vgg_base[str(size)], 3),
                         add_extras(extras[str(size)] ,1024, size=size),size),
                         mbox[str(size)], num_classes), num_classes=num_classes,size=size)
