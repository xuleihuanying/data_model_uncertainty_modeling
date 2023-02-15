# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import torch
import torch.nn as nn
import logging
from contrib import adf
import random
import os
import numpy as np
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, mode='train', learning_rate=0.001, batch_size=50, in_planes=6, planes=1, stride=1, p=0.5, keep_variance_fn=None):
        super(Model, self).__init__()

        self.mode = mode
        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.ensemble = 10

        random.seed(1)
        os.environ['PYTHONHASHSEED'] = str(1)
        np.random.seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.keep_variance_fn = keep_variance_fn

        # net = slim.conv2d(images, 32, [2, 2], scope='conv1')  # [None 64, 64, 32]
        # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')  # [None, 32,32,32]
        # net = slim.conv2d(net, 64, [2, 2], scope='conv2')  # [None, 32,32,64]
        # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  # [None, 16,16,64]
        # net = slim.conv2d(net, 128, [2, 2], scope='conv3')  # [None, 16,16,128]
        # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  # [None, 8,8,128]
        # net = slim.conv2d(net, 256, [2, 2], scope='conv4')  # [None, 8,8,256]
        # net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')  # [None, 4,4,256]
        #
        # net = slim.conv2d(net, 2048, [4, 4], padding='VALID', scope='conv5')  # [None, 1,1,2048]
        # net = slim.dropout(net, 0.5, is_training=is_training)
        # net = slim.conv2d(net, 2048, 1, padding='VALID', scope='conv6')  # [None, 1,1,1024]
        # net = slim.dropout(net, 0.5, is_training=is_training)

        self.conv1 = adf.Conv2d(7, 32, kernel_size=2, bias=False, padding = 0, keep_variance_fn=self.keep_variance_fn) #[None 64, 64, 32]
        self.relu1 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.maxpool1 = adf.MaxPool2d(keep_variance_fn=self.keep_variance_fn)

        self.conv2 = adf.Conv2d(32, 64, kernel_size=2, stride=stride, bias=False,keep_variance_fn=self.keep_variance_fn)
        self.relu2 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.maxpool2 = adf.MaxPool2d(keep_variance_fn=self.keep_variance_fn)

        self.conv3 = adf.Conv2d(64, 128, kernel_size=2, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.relu3 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.maxpool3 = adf.MaxPool2d(keep_variance_fn=self.keep_variance_fn)

        self.conv4 = adf.Conv2d(128, 256, kernel_size=2, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.relu4 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.maxpool4 = adf.MaxPool2d(keep_variance_fn=self.keep_variance_fn)

        self.conv5 = adf.Conv2d(256, 2048, kernel_size=4, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.relu5 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.dropout1 = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

        self.conv6 = adf.Conv2d(2048, 2048, kernel_size=1, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.relu6 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.dropout2 = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn) #[None, 1,1,1024]

        # decoder
        self.conv7 = adf.ConvTranspose2d(2048, 512, kernel_size=4, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.relu7 = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.dropout3 = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)  # [None, 1,1,1024]

        self.conv8 = adf.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False,
                                         keep_variance_fn=self.keep_variance_fn)
        self.conv9 = adf.Conv2d(256, 128, kernel_size=2, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)
        self.conv10 = adf.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False,
                                         keep_variance_fn=self.keep_variance_fn)
        self.conv11 = adf.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False,
                                          keep_variance_fn=self.keep_variance_fn)
        self.conv12 = adf.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False,
                                          keep_variance_fn=self.keep_variance_fn)
        self.conv13 = adf.Conv2d(16, 1, kernel_size=2, stride=stride, bias=False,
                                keep_variance_fn=self.keep_variance_fn)

    def forward(self, inputs_mean, inputs_variance):
        x = inputs_mean, inputs_variance
        x = F.pad(x[0], (0,1,0,1)), F.pad(x[1], (0,1,0,1))
        x = self.conv1(x[0],x[1])
        x = self.relu1(x[0],x[1])
        x = self.maxpool1(x[0],x[1])

        x = F.pad(x[0], (0,1,0,1)), F.pad(x[1], (0,1,0,1))
        x = self.conv2(x[0],x[1])
        x = self.relu2(x[0],x[1])
        x = self.maxpool2(x[0],x[1])

        x = F.pad(x[0], (0,1,0,1)), F.pad(x[1], (0,1,0,1))
        x = self.conv3(x[0],x[1])
        x = self.relu3(x[0],x[1])
        x = self.maxpool3(x[0],x[1])

        x = F.pad(x[0], (0,1,0,1)), F.pad(x[1], (0,1,0,1))
        x = self.conv4(x[0],x[1])
        x = self.relu4(x[0],x[1])
        x = self.maxpool4(x[0],x[1])

        # x = F.pad(x[0], (0,1,0,1)), F.pad(x[1], (0,1,0,1))
        x = self.conv5(x[0],x[1])
        x = self.relu5(x[0],x[1])
        x = self.dropout1(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv6(x[0],x[1])
        x = self.relu6(x[0],x[1])
        x = self.dropout2(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv7(x[0],x[1])
        x = self.relu7(x[0],x[1])
        x = self.dropout3(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv8(x[0],x[1])

        x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv9(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv10(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv11(x[0],x[1])

        # x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv12(x[0],x[1])

        x = F.pad(x[0], (0, 1, 0, 1)), F.pad(x[1], (0, 1, 0, 1))
        x = self.conv13(x[0],x[1])

        out = x


        # out = self.maxpool1(*self.relu1(*self.conv1(*x)))
        # out = self.maxpool2(*self.relu2(*self.conv2(*x)))
        # out = self.maxpool3(*self.relu3(*self.conv3(*x)))
        # out = self.maxpool4(*self.relu4(*self.conv4(*x)))
        # out = self.dropout1(*self.relu5(*self.conv5(*x)))
        # out = self.dropout2(*self.relu6(*self.conv6(*x)))
        # out = self.dropout3(*self.relu7(*self.conv7(*x)))

        # out = self.conv8(*x)
        # out = self.conv9(*x)
        # out = self.conv10(*x)
        # out = self.conv11(*x)
        # out = self.conv12(*x)
        # out = self.conv13(*x)

        return out

