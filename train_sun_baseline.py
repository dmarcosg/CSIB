import csv
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import SUNAttributesDataset, SoNDataset, ToTensor, Rescale, RandomCrop, VerticalFlip, my_collate, \
    Rotate, NetSUNTop, NetSoNTopSIN, FullNetGeo

gpu_no = 0  # Set to False for cpu-version
freeze_basenet_until_epoch = 5
# Directory to save the model in
net_folder = 'sun_son'
out_net_name = 'sun_base102.pt'

# Directory of the init model. init_net_name = None for starting from scratch
init_net_folder = 'sun_son'
init_net_name = None

do_plot = True


init_lr = 5e-4
reduce_lr = [10,20,30]
n_epochs = 60

if type(gpu_no) == int:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if not os.path.exists(net_folder):
    os.mkdir(net_folder)



# Read images and labels of SUN Attributes dataset
data_path = '/home/diego/Downloads/Datasets/SUN/SUNAttributeDB/'
# Get the attribute names
temp = scipy.io.loadmat(data_path+'attributes.mat')
attr_names = [m[0][0] for m in temp['attributes']]
# get the labels and the image names
temp = scipy.io.loadmat(data_path+'attributeLabels_continuous.mat')
labels = temp['labels_cv']
temp = scipy.io.loadmat(data_path+'trainval_idx.mat')
trainval_split = temp['sets']
temp = scipy.io.loadmat(data_path+'images.mat')
im_names = [m[0][0] for m in temp['images']]
# Split in train-val-test (80-10-10)
train_indeces = np.where(trainval_split==0)[0].astype(int)
val_indeces = np.where(trainval_split==1)[0].astype(int)
test_indeces = np.where(trainval_split==2)[0].astype(int)
labels_train = labels[train_indeces,:]
im_names_train = [im_names[i] for i in train_indeces]
labels_val = labels[val_indeces,:]
im_names_val = [im_names[i] for i in val_indeces]
labels_test = labels[test_indeces,:]
im_names_test = [im_names[i] for i in test_indeces]


composed = transforms.Compose([Rescale((500,500)),Rotate(5),RandomCrop(450),VerticalFlip(),ToTensor()])

dataset_sun_train = SUNAttributesDataset(data_path,im_names_train,labels_train,attr_names,tr=composed)
dataset_sun_val = SUNAttributesDataset(data_path,im_names_val,labels_val,attr_names,tr=composed)
dataset_sun_test = SUNAttributesDataset(data_path,im_names_test,labels_test,attr_names,tr=composed)

dataloader_sun_train = DataLoader(dataset_sun_train, batch_size=10,shuffle=True,num_workers=4)
dataloader_sun_val = DataLoader(dataset_sun_val, batch_size=10,shuffle=False,num_workers=4)
dataloader_sun_test = DataLoader(dataset_sun_test, batch_size=10,shuffle=False,num_workers=4)
#inspect_dataset(dataset,attr_names)


# Prepare initial model
basenet = models.resnet50(pretrained=True)
if freeze_basenet_until_epoch > 0:
    for param in basenet.parameters():
        param.requires_grad = False

if init_net_name is not None:
    net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop())
    net.load_state_dict(torch.load(os.path.join(init_net_folder,init_net_name)),strict=False)
else:
    net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop())



optimizer = optim.SGD(net.parameters(),lr=init_lr,weight_decay=0,momentum=0.9)


loss_sun = nn.BCELoss(reduction='none')
nl = nn.Tanh()
epoch_loss_sun = []
epoch_loss_val_sun = []

sun_class_mean = torch.Tensor(labels_train.mean(0)).unsqueeze(0).to(device)



iter_num = 500
iter_num_val = 100

ax = plt.axes()
pool = nn.AdaptiveAvgPool2d(1)
nl = nn.Tanh()

net = net.to(device)

for epoch in range(n_epochs):  # loop over the dataset multiple times
    ############################################################################################
    # TRAIN
    ############################################################################################
    if freeze_basenet_until_epoch == epoch:
        for param in basenet.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(net.parameters(), lr=init_lr, weight_decay=0.001, momentum=0.9)

    if np.any(np.array(reduce_lr) == epoch):
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] / 4
        init_lr = init_lr / 4

    iter_loss_sun = []
    iter_loss_son = []
    template_loss = 0
    dataloader_sun_train_for_epoch = iter(dataloader_sun_train)

    net.train()
    pbar = tqdm(total=iter_num)
    for i in range(iter_num):


        optimizer.zero_grad()

        # SUN Attributes
        data = dataloader_sun_train_for_epoch.next()
        out = net(data['image'].to(device))
        out = nl(pool(out).squeeze().squeeze())


        out_sun = loss_sun(out, (data['label'] > 0.5).float().to(device))

        out_sun_zeros = (out_sun * 0.5 )[data['label'].to(device) == 0]
        out_sun_nonzeros = (out_sun * 0.5 )[data['label'].to(device) > 0.5]
        out = out_sun_zeros.mean() + out_sun_nonzeros.mean()

        iter_loss_sun.append(out.item())


        out.backward()
        optimizer.step()


        pbar.set_description('Epoch ' + str(epoch) + '. Loss SUN:'+'%.4f'%(np.mean(iter_loss_sun)))
        pbar.update()
    pbar.close()
    epoch_loss_sun.append(np.mean(iter_loss_sun))

    torch.save(net.state_dict(), os.path.join(net_folder, out_net_name))

    ############################################################################################
    # VAL
    ############################################################################################
    net.eval()
    torch.cuda.empty_cache()
    iter_loss_sun = []
    dataloader_sun_val_for_epoch = iter(dataloader_sun_val)
    pbar = tqdm(total=iter_num_val)
    with torch.no_grad():
        for i in range(iter_num_val):
            # SUN Attributes

            data = dataloader_sun_train_for_epoch.next()
            out = net(data['image'].to(device))
            out = pool(out).squeeze().squeeze()
            out_sun = loss_sun(nl(out), (data['label'] > 0.5).float().to(device))
            out_sun_zeros = (out_sun * sun_class_mean)[data['label'].to(device) == 0]
            out_sun_nonzeros = ((out_sun * (1 - sun_class_mean)))[data['label'].to(device) > 0.5]
            out = out_sun_zeros.mean() + out_sun_nonzeros.mean()
            iter_loss_sun.append(out.item())


            pbar.set_description('Val loss SUN: ' + '%.4f' % (np.mean(iter_loss_sun)))
            pbar.update()

    pbar.close()
    epoch_loss_val_sun.append(np.mean(iter_loss_sun))

