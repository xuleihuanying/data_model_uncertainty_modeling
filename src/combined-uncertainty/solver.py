# import tensorflow as tf
# import tensorflow.contrib.slim as slim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os
# from torchsummary import summary
from contrib import adf
from model import Model
import numpy as np
import pickle
import os
import hdf5storage
import random
# from torchsummary import summary


# ~ from utils import resize_images
random.seed(1)
os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def model_loader():
    model = Model()

    return model


print('==> Building model...')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = model_loader().to(device)
# net.load_state_dict(torch.load('net_model.pkl'))
# net = torch.load('net_model.pkl')
# summary(net, (6, 64, 64))

# 需要使用device来指定网络在GPU还是CPU运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(net)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.1, last_epoch=-1)


mnist_dir='E:/hess/data/deeplearning/uncertainty/' # I:/data/deeplearning/uncertainty/
#/mnt/md0/XL/deeplearning/uncertainty/
# /data/run01/scv0379
batch_size=100
train_iter=50
ensemble=10

# class Solver(object):
#
#     def __init__(self, model, batch_size=100, train_iter=200,ensemble=20,
#                  mnist_dir='/mnt/md0/XL/deeplearning/uncertainty/', log_dir='logs',
#                  model_save_path='model', trained_model='model/model'):
#
#         self.model.model_loader()
#         # actually builds the graph
#         self.model.build_model()
#         self.ensemble = ensemble
#         self.batch_size = batch_size
#         self.train_iter = train_iter
#         self.mnist_dir = mnist_dir
#         self.log_dir = log_dir
#         self.model_save_path = model_save_path
#         self.trained_model = model_save_path + '/model'
# self.config = tf.ConfigProto()
# self.config.gpu_options.allow_growth = True
# self.config.allow_soft_placement = True


# tf.set_random_seed(1)


def load_train(path, split='train'):
    print('[*] Loading training and validating dataset.')

    train_data = hdf5storage.loadmat(path + 'train_data.mat')
    train_data_label = hdf5storage.loadmat(path + 'train_data_label.mat')
    train_data = train_data['train_data']  # 29*18*time*sample
    train_data_label = train_data_label['train_data_label']  # 29*18*time*sample

    train_data_std = hdf5storage.loadmat(path + 'train_data_std.mat')
    train_data_label_std = hdf5storage.loadmat(path + 'train_data_label_std.mat')
    train_data_std = train_data_std['train_data_std']  # 29*18*time*sample
    train_data_label_std = train_data_label_std['train_data_label_std']  # 29*18*time*sample

    train_data = np.transpose(train_data, (3, 2, 0, 1))  # sample*channel*width*height*
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2],
                                    train_data.shape[3])
    train_data_label = np.transpose(train_data_label, (3, 2, 0, 1))
    train_data_label = train_data_label.reshape(train_data_label.shape[0], train_data_label.shape[1],
                                                train_data_label.shape[2], train_data_label.shape[3])

    train_data_std = np.transpose(train_data_std, (3, 2, 0, 1))  # sample*width*height*channel
    train_data_std = train_data_std.reshape(train_data_std.shape[0], train_data_std.shape[1], train_data_std.shape[2],
                                            train_data_std.shape[3])

    train_data_label_std = np.transpose(train_data_label_std, (3, 2, 0, 1))  # sample*width*height*channel
    train_data_label_std = train_data_label_std.reshape(train_data_label_std.shape[0], train_data_label_std.shape[1],
                                                        train_data_label_std.shape[2],
                                                        train_data_label_std.shape[3])

    validate_data = hdf5storage.loadmat(path + 'validate_data.mat')
    validate_data_label = hdf5storage.loadmat(path + 'validate_data_label.mat')
    validate_data = validate_data['validate_data']
    validate_data_label = validate_data_label['validate_data_label']

    validate_data_std = hdf5storage.loadmat(path + 'validate_data_std.mat')
    validate_data_label_std = hdf5storage.loadmat(path + 'validate_data_label_std.mat')
    validate_data_std = validate_data_std['validate_data_std']  # 29*18*time*sample
    validate_data_label_std = validate_data_label_std['validate_data_label_std']  # 29*18*time*sample

    validate_data = np.transpose(validate_data, (3, 2, 0, 1))
    validate_data = validate_data.reshape(validate_data.shape[0], validate_data.shape[1], validate_data.shape[2],
                                          validate_data.shape[3])
    validate_data_label = np.transpose(validate_data_label, (3, 2, 0, 1))
    validate_data_label = validate_data_label.reshape(validate_data_label.shape[0], validate_data_label.shape[1],
                                                      validate_data_label.shape[2], validate_data_label.shape[3])

    validate_data_std = np.transpose(validate_data_std, (3, 2, 0, 1))  # sample*width*height*channel
    validate_data_std = validate_data_std.reshape(validate_data_std.shape[0], validate_data_std.shape[1],
                                                  validate_data_std.shape[2],
                                                  validate_data_std.shape[3])

    validate_data_label_std = np.transpose(validate_data_label_std, (3, 2, 0, 1))  # sample*width*height*channel
    validate_data_label_std = validate_data_label_std.reshape(validate_data_label_std.shape[0],
                                                              validate_data_label_std.shape[1],
                                                              validate_data_label_std.shape[2],
                                                              validate_data_label_std.shape[3])

    return train_data, train_data_label, train_data_std, train_data_label_std, validate_data, validate_data_label, validate_data_std, validate_data_label_std


def load_test(path, split='test'):
    test_data = hdf5storage.loadmat(path + 'test_data.mat')
    test_data_label = hdf5storage.loadmat(path + 'test_data_label.mat')
    test_data = test_data['test_data']
    test_data_label = test_data_label['test_data_label']

    test_data = np.transpose(test_data, (3, 2, 0, 1))
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2],
                                  test_data.shape[3])
    test_data_label = np.transpose(test_data_label, (3, 2, 0, 1))
    test_data_label = test_data_label.reshape(test_data_label.shape[0], test_data_label.shape[1],
                                              test_data_label.shape[2], test_data_label.shape[3])

    test_data_std = hdf5storage.loadmat(path + 'test_data_std.mat')
    test_data_label_std = hdf5storage.loadmat(path + 'test_data_label_std.mat')
    test_data_std = test_data_std['test_data_std']
    test_data_label_std = test_data_label_std['test_data_label_std']

    test_data_std = np.transpose(test_data_std, (3, 2, 0, 1))
    test_data_std = test_data_std.reshape(test_data_std.shape[0], test_data_std.shape[1], test_data_std.shape[2],
                                          test_data_std.shape[3])
    test_data_label_std = np.transpose(test_data_label_std, (3, 2, 0, 1))
    test_data_label_std = test_data_label_std.reshape(test_data_label_std.shape[0], test_data_label_std.shape[1],
                                                      test_data_label_std.shape[2], test_data_label_std.shape[3])

    return test_data, test_data_label, test_data_std, test_data_label_std



class Loss_function(torch.nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()


    def forward(self, outputs, outputs_var, targets, targets_var, eps=1e-5):

        # 根据不确定性赋权，y的不确定性大的地方赋予较大的权重
        var_total = torch.mean(torch.mean( outputs_var, dim=1), dim=0) # 64*64

        # targets_var_en = torch.reshape(targets_var, [targets_var.shape[0], targets_var.shape[1], targets_var.shape[2],
        #               targets_var.shape[3], 1]).expand(-1,-1,-1,-1, ensemble)
        # targets_en = torch.reshape(targets, [targets.shape[0], targets.shape[1], targets.shape[2],
        #               targets.shape[3], 1]).expand(-1,-1,-1,-1, ensemble)
        # targets_en = targets_en + torch.sqrt(targets_var_en)*torch.randn(targets_var_en.shape).to(device)
        # targets_en[targets_en < 0] = 0
        #
        # outputs_var_en = torch.reshape(outputs_var, [outputs_var.shape[0], outputs_var.shape[1], outputs_var.shape[2],
        #                                outputs_var.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # outputs_en = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], outputs.shape[2],
        #                            outputs.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # outputs_en = outputs_en + torch.sqrt(outputs_var_en) * torch.randn(outputs_var_en.shape).to(device)
        # outputs_en[outputs_en < 0] = 0
        # loss = torch.mean(0.5 * (
        #         torch.mean(torch.mean(torch.mean(torch.square(targets_en - outputs_en), dim=4), dim=1), dim=0) / (
        #         var_total + 1e-3) + torch.log(var_total + 1e-3)))
        # w = torch.mean(torch.mean(targets_var, dim=1),dim=0)
        # w = w / torch.sum(w)

        loss = torch.mean(0.5 * (
                torch.mean(torch.mean(torch.square(targets - outputs), dim=1), dim=0) / (
                var_total + 1e-3) + torch.log(var_total + 1e-3)) )
        # loss = torch.mean( torch.square(targets - outputs) / (outputs_var+1e-3) + torch.log(outputs_var+1e-3) )

        # var_total = torch.mean(torch.mean(outputs_var, dim=1), dim=0)  # 64*64
        # # targets_var_en = torch.reshape(targets_var, [targets_var.shape[0], targets_var.shape[1], targets_var.shape[2],
        # #                                              targets_var.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # targets_en = torch.reshape(targets, [targets.shape[0], targets.shape[1], targets.shape[2],
        #                                      targets.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # # targets_en = targets_en + torch.sqrt(targets_var_en)*torch.randn(targets_var_en.shape).to(device)
        # # targets_en[targets_en < 0] = 0
        # # targets_en, indices = torch.sort(targets_en, dim=4, descending=False)
        #
        # outputs_var_en = torch.reshape(outputs_var, [outputs_var.shape[0], outputs_var.shape[1], outputs_var.shape[2],
        #                                              outputs_var.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # outputs_en = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], outputs.shape[2],
        #                                      outputs.shape[3], 1]).expand(-1, -1, -1, -1, ensemble)
        # outputs_en = outputs_en + torch.sqrt(outputs_var_en) * torch.randn(outputs_var_en.shape).to(device)
        # outputs_en[outputs_en < 0] = 0
        # # outputs_en, indices = torch.sort(outputs_en, dim=4, descending=False)
        #
        # loss = torch.mean(0.5 * (
        #         torch.mean(torch.mean(torch.mean(torch.square(targets_en - outputs_en), dim=4), dim=1), dim=0) / (
        #         var_total + 1e-3) + torch.log(var_total + 1e-3)))

        # var_total = torch.mean(torch.mean(outputs_var, dim=1), dim=0)
        # loss = torch.mean(0.5 * (
        #         torch.mean(torch.mean(torch.square(targets - outputs_mean), dim=1), dim=0) / (
        #         var_total + 1e-3) + torch.log(var_total + 1e-3)))
        return loss

criterion = Loss_function()


def train(net):
    # make directory if not exists
    # if tf.gfile.Exists(self.log_dir):
    #     tf.gfile.DeleteRecursively(self.log_dir)
    # tf.gfile.MakeDirs(self.log_dir)
    random.seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    print('[*] Training.')
    net.train()
    train_val_loss_save = np.empty([train_iter, 2], dtype=float) * np.nan  # save to txt

    # images, labels = self.load_mnist(self.mnist_dir, split='train')
    train_data, train_data_label, train_data_std, train_data_label_std, validate_data, validate_data_label, validate_data_std, validate_data_label_std = load_train(
        mnist_dir)

    # with tf.Session(config=self.config) as sess:
    #     tf.global_variables_initializer().run()
    #     saver = tf.train.Saver()

        # summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

    print('Start training.')
    count = 0
    t = 0
    min_loss = 1e+10
    for step in range(train_iter):
        # 在train stage进行dropout，在test stage不进行dropout，但是需要对输入数据加入一定的噪声
        count += 1
        t += 1

        num_batch = int(train_data.shape[0] / batch_size)
        if (train_data.shape[0] % batch_size) > 0:
            num_batch = num_batch + 1
        train_loss_total = 0
        # self.model.mode = 'train'
        for index_batch in range(num_batch):

            if index_batch < num_batch - 1:
                inputs = train_data[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
                targets = train_data_label[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
                inputs_std = train_data_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
                targets_std = train_data_label_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
            else:
                inputs = train_data[index_batch * batch_size:train_data.shape[0], :, :, :]
                targets = train_data_label[index_batch * batch_size:train_data_label.shape[0], :, :,
                          :]
                inputs_std = train_data_std[index_batch * batch_size:train_data_label.shape[0], :, :,
                             :]
                targets_std = train_data_label_std[
                              index_batch * batch_size:train_data_label_std.shape[0], :, :, :]

            optimizer.zero_grad()

            # outputs = net(torch.from_numpy(inputs).float().to(device), torch.from_numpy(inputs_std * inputs_std).float().to(device))
            # outputs2 = net(torch.from_numpy(inputs).float().to(device),
            #               torch.from_numpy(inputs_std * inputs_std).float().to(device))
            # outputs_mean, outputs_var = outputs
            # outputs_mean2, outputs_var2 = outputs2

            for i in range(ensemble):
                o_mean, o_var = net(torch.from_numpy(inputs).float().to(device),
                                    torch.from_numpy(inputs_std * inputs_std).float().to(device))
                if i==0:
                    outputs_mean = o_mean
                    outputs_var = o_var
                else:
                    outputs_mean += o_mean
                    outputs_var += o_var
            outputs_mean = outputs_mean / ensemble
            outputs_var = outputs_var / ensemble

            loss = criterion(outputs_mean, outputs_var, torch.from_numpy(targets).float().to(device), torch.from_numpy(targets_std * targets_std).float().to(device))
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        # sess.run(self.model.train_op, feed_dict)
        # train_loss = sess.run([self.model.loss], feed_dict)
        train_loss_total = train_loss_total / train_data.shape[0]
        # if step==0:
        #     saver.save(sess, os.path.join(self.model_save_path, 'model'))
        print('Step: [%d]  train loss: [%.6f] ' \
              % (step, train_loss_total))

        # calculate validate loss
        # val_loss_total = 0
        # val_loss_total_space = 0
        # self.model.mode = 'test'
        # pre_val_total = torch.empty([validate_data_label.shape[0], validate_data_label.shape[1], validate_data_label.shape[2], \
        #                              validate_data_label.shape[3], ensemble], dtype=torch.float)  # save to txt
        # pre_val_total_var = torch.empty([validate_data_label.shape[0], validate_data_label.shape[1], validate_data_label.shape[2], \
        #                                  validate_data_label.shape[3], ensemble], dtype=torch.float)  # save to txt
        # label_total = np.empty([validate_data_label.shape[0], validate_data_label.shape[1],validate_data_label.shape[2], \
        #                           self.ensemble], dtype=float) * np.nan  # save to txt

        inputs = validate_data
        targets = validate_data_label
        inputs_std = validate_data_std
        targets_std = validate_data_label_std
        with torch.no_grad():
            outputs = net(torch.from_numpy(inputs).float().to(device),
                          torch.from_numpy(inputs_std * inputs_std).float().to(device))
            outputs_mean, outputs_var = outputs  # [100 1 64 64]
            outputs_mean[outputs_mean < 0] = 0
            loss = criterion(outputs_mean, outputs_var,torch.from_numpy(targets).float().to(device),torch.from_numpy(targets_std * targets_std).float().to(device))
            val_loss_total = loss.item()

        val_loss_total = val_loss_total / validate_data.shape[0]
        val_loss_total_ensemble = val_loss_total


        # val_loss_total_ensemble = 0
        # for index_ensem in range(ensemble):
        #     # validate_data_with_un = np.random.randn(validate_data.shape[0], validate_data.shape[1],
        #     #                                         validate_data.shape[2], \
        #     #                                         validate_data.shape[3]) * validate_data_std + validate_data
        #     # validate_data_with_un[validate_data_with_un < 0] = 0
        #     num_batch = int(validate_data.shape[0] / batch_size)
        #     if (validate_data.shape[0] % batch_size) > 0:
        #         num_batch = num_batch + 1
        #     # self.model.mode = 'train'
        #     val_loss_total = 0
        #     for index_batch in range(num_batch):
        #
        #         if index_batch < num_batch - 1:
        #             inputs = validate_data[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
        #             targets = validate_data_label[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
        #             inputs_std = validate_data_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
        #             targets_std = validate_data_label_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
        #         else:
        #             inputs = validate_data[index_batch * batch_size:validate_data.shape[0], :, :, :]
        #             targets = validate_data_label[index_batch * batch_size:validate_data_label.shape[0], :, :,
        #                       :]
        #             inputs_std = validate_data_std[index_batch * batch_size:validate_data_std.shape[0], :, :,
        #                          :]
        #             targets_std = validate_data_label_std[
        #                           index_batch * batch_size:validate_data_label_std.shape[0], :, :, :]
        #         with torch.no_grad():
        #             outputs = net(torch.from_numpy(inputs).float().to(device), torch.from_numpy(inputs_std * inputs_std).float().to(device))
        #             outputs_mean, outputs_var = outputs #[100 1 64 64]
        #             outputs_mean[outputs_mean < 0] = 0
        #             loss = criterion(outputs_mean, outputs_var,torch.from_numpy(targets).float().to(device),torch.from_numpy(targets_std * targets_std).float().to(device))
        #             val_loss_total += loss.item()
        #         if index_batch < num_batch - 1:
        #             pre_val_total[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :, index_ensem] = outputs_mean[:, :, :, :]
        #             pre_val_total_var[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :, index_ensem] = outputs_var[:, :, :, :]
        #         else:
        #             pre_val_total[index_batch * batch_size:validate_data.shape[0], :, :, :, index_ensem] = outputs_mean[:, :, :, :]
        #             pre_val_total_var[index_batch * batch_size:validate_data.shape[0], :, :, :, index_ensem] = outputs_var[:, :, :, :]
        #     val_loss_total = val_loss_total / validate_data.shape[0]
        #     val_loss_total_ensemble = val_loss_total_ensemble + val_loss_total
        # val_loss_total_ensemble = val_loss_total_ensemble / ensemble

        pre_val_total_mean = outputs_mean
        pre_val_total_std = torch.sqrt(outputs_var)

        # pre_val_total_mean = torch.mean(pre_val_total, dim=4)

        # pre_val_total_mean = torch.reshape(pre_val_total_mean,
        #                                 [pre_val_total_mean.shape[0], pre_val_total_mean.shape[1],
        #                                  pre_val_total_mean.shape[2], 1])
        # pre_val_total_var = torch.mean(pre_val_total_var, dim=4)
        # pre_val_total_var = torch.reshape(pre_val_total_var,
        #                                 [pre_val_total_var.shape[0], pre_val_total_var.shape[1],
        #                                  pre_val_total_var.shape[2], 1])

        # loss = criterion(pre_val_total_mean, pre_val_total_var, torch.from_numpy(targets).float().to(device), torch.from_numpy(targets_std * targets_std).float().to(device) )
        # pre_var = np.mean(
        #     np.mean(np.square(pre_val_total - np.repeat(pre_val_total_mean, self.ensemble, axis=3)),
        #             axis=3), axis=0)

        # pre_val_total_std = torch.sqrt(
        #     torch.mean(pre_val_total * pre_val_total, dim=4) - torch.square(
        #         torch.mean(pre_val_total, dim=4)) + torch.mean(
        #         pre_val_total_var, dim=4))



        # pre_val_total_std = torch.sqrt(pre_val_total_var)
        # pre_val_total_std = np.sqrt(0.5 * (np.square(validate_data_label[:,:,:,0] - pre_val_total_mean) / np.square(sigma + 1e-3) + np.log(np.square(sigma))) )
        # del pre_val_total
        pre_val_total_mean = torch.reshape(pre_val_total_mean, [pre_val_total_mean.shape[0],
                                                        pre_val_total_mean.shape[2] * pre_val_total_mean.shape[
                                                            3]])
        pre_val_total_std = torch.reshape(pre_val_total_std, [pre_val_total_std.shape[0],
                                                      pre_val_total_std.shape[2] * pre_val_total_std.shape[3]])
        print('Step: [%d]  val loss: [ %.6f] ' \
              % (step, val_loss_total_ensemble))

        ### 保存train_loss, val_loss，保存预测均值与预测方差
        train_val_loss_save[step, 0] = train_loss_total
        train_val_loss_save[step, 1] = val_loss_total_ensemble

        if (val_loss_total_ensemble < min_loss):  # and (step%20==0 or step==0):
            min_loss = val_loss_total_ensemble
            torch.save(net, 'net_model.pkl')
            # torch.save(net.state_dict(), 'net_model.pkl')
            # np.savetxt('val_predict_mean.txt', pre_val_total_mean.detach().cpu().numpy())
            # np.savetxt('val_predict_std.txt', pre_val_total_std.detach().cpu().numpy())

        # scheduler.step()
    np.savetxt('train_val_loss.txt', train_val_loss_save)


def test(net):
    random.seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('[*] Test.')
    net = torch.load('net_model.pkl')
    # net = model_loader().to(device)
    # net.load_state_dict(torch.load('net_model.pkl'))
    validate_data, validate_data_label, validate_data_std, validate_data_label_std = load_test(mnist_dir)

    # pre_val_total = torch.empty([validate_data_label.shape[0], validate_data_label.shape[1], validate_data_label.shape[2], \
    #                              validate_data_label.shape[3], ensemble], dtype=torch.float)  # save to txt
    # pre_val_total_var = torch.empty([validate_data_label.shape[0], validate_data_label.shape[1], validate_data_label.shape[2], \
    #                                  validate_data_label.shape[3], ensemble], dtype=torch.float)   # save to txt
    # label_total = np.empty([validate_data_label.shape[0], validate_data_label.shape[1],validate_data_label.shape[2], \
    #                           self.ensemble], dtype=float) * np.nan  # save to txt

    inputs = validate_data
    targets = validate_data_label
    inputs_std = validate_data_std
    targets_std = validate_data_label_std
    with torch.no_grad():
        outputs = net(torch.from_numpy(inputs).float().to(device),
                      torch.from_numpy(inputs_std * inputs_std).float().to(device))
        outputs_mean, outputs_var = outputs  # [100 1 64 64]
        outputs_mean[outputs_mean < 0] = 0
        loss = criterion(outputs_mean, outputs_var, torch.from_numpy(targets).float().to(device),
                         torch.from_numpy(targets_std * targets_std).float().to(device))
        val_loss_total = loss.item()

    val_loss_total = val_loss_total / validate_data.shape[0]
    # val_loss_total_ensemble = val_loss_total


    # for index_ensem in range(ensemble):
    #     # validate_data_with_un = np.random.randn(validate_data.shape[0], validate_data.shape[1],
    #     #                                         validate_data.shape[2], \
    #     #                                         validate_data.shape[3]) * validate_data_std + validate_data
    #     # validate_data_with_un[validate_data_with_un < 0] = 0
    #     # inputs = validate_data
    #     # inputs_std = validate_data_std
    #
    #     num_batch = int(validate_data.shape[0] / batch_size)
    #     if (validate_data.shape[0] % batch_size) > 0:
    #         num_batch = num_batch + 1
    #     # self.model.mode = 'train'
    #     # val_loss_total = 0
    #     for index_batch in range(num_batch):
    #
    #         if index_batch < num_batch - 1:
    #             inputs = validate_data[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
    #             # targets = validate_data_label[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
    #             inputs_std = validate_data_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
    #             # targets_std = validate_data_label_std[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :]
    #         else:
    #             inputs = validate_data[index_batch * batch_size:validate_data.shape[0], :, :, :]
    #             # targets = validate_data_label[index_batch * batch_size:validate_data_label.shape[0], :, :,
    #             #           :]
    #             inputs_std = validate_data_std[index_batch * batch_size:validate_data_label.shape[0], :, :,
    #                          :]
    #             # targets_std = validate_data_label_std[
    #             #               index_batch * batch_size:validate_data_label_std.shape[0], :, :, :]
    #         with torch.no_grad():
    #             outputs = net(torch.from_numpy(inputs).float().to(device),
    #                           torch.from_numpy(inputs_std * inputs_std).float().to(device))
    #             outputs_mean, outputs_var = outputs  # [100 1 64 64]
    #             outputs_mean[outputs_mean < 0] = 0
    #
    #         if index_batch < num_batch - 1:
    #             pre_val_total[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :,
    #             index_ensem] = outputs_mean[:, :, :, :]
    #             pre_val_total_var[index_batch * batch_size:(index_batch + 1) * batch_size, :, :, :,
    #             index_ensem] = outputs_var[:, :, :, :]
    #         else:
    #             pre_val_total[index_batch * batch_size:validate_data.shape[0], :, :, :, index_ensem] = outputs_mean[:,
    #                                                                                                    :, :, :]
    #             pre_val_total_var[index_batch * batch_size:validate_data.shape[0], :, :, :, index_ensem] = outputs_var[
    #                                                                                                        :, :, :, :]

        # outputs = net(inputs, inputs_std * inputs_std)

        # outputs = net(torch.from_numpy(inputs).float().to(device), torch.from_numpy(inputs_std * inputs_std).float().to(device))
        # outputs_mean, outputs_var = outputs
        #
        # # pre_val = sess.run(self.model.pre_val, feed_dict)
        # outputs_mean[outputs_mean < 0] = 0
        # pre_val_total[:, :, :,:, index_ensem] = outputs_mean[:, :, :, :]
        # pre_val_total_var[:, :, :,:, index_ensem] = outputs_var[:, :, :, :]

    pre_val_total_mean = outputs_mean
    pre_val_total_std = torch.sqrt(outputs_var)
    # pre_val_total_std = torch.sqrt(
    #     torch.mean(pre_val_total * pre_val_total, dim=4) - torch.square(torch.mean(pre_val_total, dim=4)) + torch.mean(
    #         pre_val_total_var, dim=4))
    # pre_val_total_var = torch.mean(pre_val_total_var, dim=4)


    # pre_val_total_std = torch.sqrt(pre_val_total_var)
    # pre_val_total_std = np.sqrt(0.5 * (np.square(validate_data_label[:,:,:,0] - pre_val_total_mean) / np.square(sigma + 1e-3) + np.log(np.square(sigma))) )
    # del pre_val_total
    pre_val_total_mean = torch.reshape(pre_val_total_mean,[pre_val_total_mean.shape[0],
                                                    pre_val_total_mean.shape[2] * pre_val_total_mean.shape[
                                                        3]])
    pre_val_total_std = torch.reshape(pre_val_total_std, [pre_val_total_std.shape[0],
                                                  pre_val_total_std.shape[2] * pre_val_total_std.shape[3]])
    np.savetxt('test_predict_mean.txt', pre_val_total_mean.detach().cpu().numpy())
    np.savetxt('test_predict_std.txt', pre_val_total_std.detach().cpu().numpy())


train(net)
test(net)

