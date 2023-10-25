import torch
import torch.nn as nn
import numpy as np
import os
from sklearn import metrics
from skimage import color
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import init


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, num_step, power):
    lr = lr_poly(learning_rate, i_iter, num_step, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer_D, learning_rate_D, i_iter, num_step, power):
    lr = lr_poly(learning_rate_D, i_iter, num_step, power)
    optimizer_D.param_groups[0]['lr'] = lr
    if len(optimizer_D.param_groups) > 1:
        optimizer_D.param_groups[1]['lr'] = lr * 10


# def init_weights(net, init_type='normal'):
#     #print('initialization method [%s]' % init_type)
#     if init_type == 'kaiming':
#         net.apply(weights_init_kaiming)
#     else:
#         raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('BatchNorm') != -1:
#         init.normal_(m.weight.data, 1.0, 0.02)
#         init.constant_(m.bias.data, 0.0)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                # m.bias.data.fill_(1e-4)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
            elif isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight.data, nonlinearity='relu')
                    elif isinstance(mm, nn.BatchNorm2d):
                        mm.weight.data.fill_(1.)
                        # mm.bias.data.fill_(1e-4)
                        mm.bias.data.zero_()
                    elif isinstance(mm, nn.Linear):
                        mm.weight.data.normal_(0.0, 0.0001)
                        mm.bias.data.zero_()


def initialize_weights_model(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.0001)
        m.bias.data.zero_()
    elif isinstance(m, nn.ModuleList) or isinstance(m, nn.Sequential):
        for model in m:
            if isinstance(model, nn.Conv2d):
                nn.init.kaiming_normal_(model.weight.data, nonlinearity='relu')
            elif isinstance(model, nn.BatchNorm2d):
                model.weight.data.fill_(1.)
                model.bias.data.fill_(1e-4)
            elif isinstance(model, nn.Linear):
                model.weight.data.normal_(0.0, 0.0001)
                model.bias.data.zero_()



