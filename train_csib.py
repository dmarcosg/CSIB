import csv
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import SUNAttributesDataset, SoNDataset, ToTensor, Rescale, RandomCrop, VerticalFlip, my_collate, \
    Rotate, NetSUNTop, NetSoNTopSINReg, FullNetGeo, projection_simplex_sort, setup_model

gpu_no = 0  # Set to False for cpu-version

# Directory to save the model in
net_folder = 'sun_son'
out_net_name = 'sun_son_siam.pt'

# Directory of the init model. init_net_name = None for starting from scratch
init_net_folder = 'sun_son'
init_net_name = 'sun_base102.pt'

do_plot = False

data_path_son = '/home/diego/Downloads/Datasets/SoN_old/'
image_folders = ['images1','images2','images3','images4','images5','images6']

freeze_basenet_until_epoch = 0
epochs_only_sun = 0
epochs_only_son = freeze_basenet_until_epoch
init_lr = 2e-3
reduce_lr = [25,50,75]
n_epochs = 125
weight_h = 0.01
weight_son = 0.1
n_groups = 0
batch_size = 12

n_attributes = 102

if type(gpu_no) == int:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


if not os.path.exists(net_folder):
    os.mkdir(net_folder)

# Read images and labels of ScenicOrNot (SoN) dataset

im_paths = []
son_avg = []
son_var = []
son_lat = []
son_lon = []
with open(os.path.join(data_path_son,'SoN_Votes.csv'), 'r') as csvfile:
     SoN_reader = csv.reader(csvfile, delimiter=',')
     for row in SoN_reader:
         for image_folder in image_folders:
             im_path = os.path.join(data_path_son,image_folder,row[0]+'.jpg')
             if os.path.isfile(im_path):
                im_paths.append(im_path)
                son_avg.append(np.float32(row[3]))
                son_var.append(np.float32(row[4]))
                son_lat.append(np.float32(row[1]))
                son_lon.append(np.float32(row[2]))

np.random.seed(11)
labels_son = np.array([son_avg]).transpose()
loc_son = np.array([son_lat,son_lon]).transpose()
random_idx = np.arange(len(loc_son))
np.random.shuffle(random_idx)
init_loc = loc_son[random_idx[0:200],:]


# Build train-val-test sets
im_paths_son_train = im_paths[0:150000]
labels_son_train = labels_son[0:150000,:]
im_paths_son_val = im_paths[150000:155000]
labels_son_val = labels_son[150000:155000,:]
im_paths_son_test = im_paths[155000:160000]
labels_son_test = labels_son[155000:160000,:]

composed = transforms.Compose([Rescale((500,500)),VerticalFlip(),ToTensor()])
dataset_son_train = SoNDataset(im_paths_son_train,labels_son_train,loc_son,tr=composed)
dataset_son_val = SoNDataset(im_paths_son_val,labels_son_val,loc_son,tr=composed)
dataset_son_test = SoNDataset(im_paths_son_test,labels_son_test,loc_son,tr=composed)

dataloader_son_train = DataLoader(dataset_son_train, batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=my_collate)
dataloader_son_val = DataLoader(dataset_son_val, batch_size=batch_size,shuffle=False,num_workers=4,collate_fn=my_collate)
dataloader_son_test = DataLoader(dataset_son_test, batch_size=batch_size,shuffle=False,num_workers=4,collate_fn=my_collate)


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
sun_class_mean = torch.Tensor(labels_train.mean(0)).unsqueeze(0).to(device)




dataset_sun_train = SUNAttributesDataset(data_path,im_names_train,labels_train,attr_names,tr=composed)
dataset_sun_val = SUNAttributesDataset(data_path,im_names_val,labels_val,attr_names,tr=composed)
dataset_sun_test = SUNAttributesDataset(data_path,im_names_test,labels_test,attr_names,tr=composed)

dataloader_sun_train = DataLoader(dataset_sun_train, batch_size=11,shuffle=True,num_workers=4)
dataloader_sun_val = DataLoader(dataset_sun_val, batch_size=11,shuffle=False,num_workers=4)
dataloader_sun_test = DataLoader(dataset_sun_test, batch_size=11,shuffle=False,num_workers=4)
#inspect_dataset(dataset,attr_names)


net = setup_model(init_net_name=init_net_name,
                  init_net_folder=init_net_folder,
                  regression_avg=dataset_son_train.label_avg,
                  n_attributes=n_attributes,
                  n_groups=n_groups)
net = net.to(device)

for param in net.basenet.parameters():
    param.requires_grad = False

optimizer = optim.SGD([
    {'params': list(net.topnet.fc1.parameters()), 'lr': 1*init_lr},
    {'params': list(net.topnet.fc2.parameters()), 'lr': 1 * init_lr, 'weight_decay':0.00}],lr=init_lr,weight_decay=0.00,momentum=0.9)


loss_sun = nn.BCELoss(reduction='none')
loss_son = nn.MSELoss(reduction='none')
loss_level = nn.MSELoss(reduction='none')
nl = nn.Tanh()
epoch_loss_sun = []
epoch_loss_val_sun = []
epoch_loss_son = []
epoch_loss_val_son = []
epoch_loss_h = []
epoch_loss_val_h = []



iter_num = 400
iter_num_val = 10



for epoch in range(n_epochs):  # loop over the dataset multiple times
    ############################################################################################
    # TRAIN
    ############################################################################################

    if epochs_only_son == epoch:
        for param in list(net.basenet.parameters())[-2:]:
            param.requires_grad = True
        optimizer = optim.SGD([
            {'params': list(net.topnet.fc1.parameters()), 'lr': 1*init_lr,'weight_decay':0.0},
            {'params': list(net.topnet.fc2.parameters()), 'lr': 1*init_lr,'weight_decay':0.00},
            {'params': list(net.basenet.parameters())[-2:]}], lr=init_lr, weight_decay=0.00,momentum=0.9)

    if freeze_basenet_until_epoch == epoch:
        for param in net.basenet.parameters():
            param.requires_grad = True
        optimizer = optim.SGD([
            {'params': list(net.topnet.fc1.parameters()), 'lr': 1*init_lr,'weight_decay':0.0},
            {'params': list(net.topnet.fc2.parameters()), 'lr': 1* init_lr, 'weight_decay':0.00},
            {'params': net.basenet.parameters(), 'lr':0.1*init_lr}], lr=init_lr, weight_decay=0.00,momentum=0.9)

    if np.any(np.array(reduce_lr) == epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 4
        init_lr = init_lr / 4


    if epoch_start_sparse == epoch:
        net.topnet.topks = [1,2,3,4,5,6,7,8]
    iter_loss_sun = []
    iter_loss_son = []
    iter_loss_son_mean = []
    iter_loss_h = []
    iter_loss_son_level = [[] for i in range(1+len(net.topnet.topks))]
    template_loss = 0
    dataloader_sun_train_for_epoch = iter(dataloader_sun_train)
    dataloader_son_train_for_epoch = iter(dataloader_son_train)

    net.train()
    done_with_sun = False
    done_with_son = False
    pbar = tqdm(total=iter_num)
    for i in range(iter_num):



        if i%20 == 0 and do_plot:
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            #ax.scatter(init_loc[:,1],init_loc[:,0])
            #ax.scatter(net.topnet.dist.weight.data.cpu().detach().numpy()[:,1],net.topnet.dist.weight.data.cpu().detach().numpy()[:,0])
            fc1 = ax1.imshow((net.topnet.fc1.weight.data).cpu().detach().numpy().T, cmap='seismic', vmin=0, vmax=0.4)
            ax1.set_title((net.topnet.fc1.weight.data.cpu().detach().abs().numpy() < 1e-4).mean())
            #plt.colorbar(fc1, ax=ax1)
            vals = net.topnet.fc2.weight.data.cpu().detach().numpy().squeeze()# - net.topnet.fc2.weight_neg.data.cpu().detach().numpy().squeeze()
            ax2.barh(np.arange(len(vals)),vals)
            ax2.set_title((net.topnet.fc2.weight.data.cpu().detach().abs().numpy() < 1e-2).mean())
            plt.pause(0.01)
            ax1.clear()
            ax2.clear()
            plt.clf()

        optimizer.zero_grad()



        # SUN Attributes
        if not done_with_sun and (epoch>=epochs_only_son):
            try:
                data = dataloader_sun_train_for_epoch.next()

            except:
                print('SUN exhausted in train')
                done_with_sun = True
                continue
            out = net(data['image'].to(device))
            out_sun = out['x_sun']
            maps = out['maps']
            out_sun = loss_sun(nl(out_sun), (data['label'].to(device) > 0).float())
            out_sun_zeros = (out_sun)[data['label'].to(device) == 0]
            out_sun_nonzeros = ((out_sun ))[data['label'].to(device) > 0.5]
            out_sun = out_sun_zeros.mean() + out_sun_nonzeros.mean()
            iter_loss_sun.append(out_sun.item())



        # SoN
        if not done_with_son and (epoch>=epochs_only_sun):
            try:
                data = dataloader_son_train_for_epoch.next()
            except:
                print('SoN exhausted in train')
                done_with_son = True
                continue
            out = net(data['image'].to(device))
            out_son = out['x_son']

            out_groups = out['x_groups_log']
            out_groups0 = out_groups.softmax(0)
            out_groups1 = out_groups.softmax(1)
            out_groupsj = out_groups.exp()/out_groups.exp().sum()
            groups_h0 = -out_groupsj*torch.log(out_groupsj/(out_groupsj.sum(0,keepdim=True)))
            groups_h1 = -out_groupsj * torch.log(out_groupsj / (out_groupsj.sum(1, keepdim=True)))
            #mi = torch.log(out_groupsj/(out_groupsj.sum(0,keepdim=True)*out_groupsj.sum(1,keepdim=True)))
            #out_h = MI.mean()*0.1
            #out_groups_p = out_groups.exp()
            p_joint = torch.matmul(out_groups0.T,out_groups0)
            p = out_groups0.mean(0,keepdim=True)
            pp = torch.matmul(p.T,p)
            cond_h0 = torch.matmul(out_groups0.T,out_groups0.log())
            cond_h0 = cond_h0 * (1-torch.eye(cond_h0.shape[0]).type_as(cond_h0))
            cond_h1 = torch.matmul(out_groups1, out_groups1.log().T)
            cond_h1 = cond_h1 * (1 - torch.eye(cond_h1.shape[0]).type_as(cond_h1))

            groups_h = - (out_groups0+out_groups1) * (out_groups0*out_groups1).log()

            extreme_vals =  out_groups.min(0)[0].exp().max() - out_groups.max(0)[0].mean() - out_groups.max(1)[0].mean() #- torch.cat([out_groups.topk(k,dim=1)[0] for k in range(6)],dim=1).mean() #((2*out_groups.exp()-out_groups.exp().max())**2).mean()

            out_h = cond_h0.mean()+cond_h1.mean()/10+ extreme_vals*1

            iter_loss_h.append(out_h.item())
            maps = out['maps']
            out_son_mean = []
            for k in range(len(out_son)):
                out_son[k] = loss_son(out_son[k], data['label'].to(device))
                iter_loss_son_level[k].append(out_son[k].mean().item())

            out_son = torch.stack(out_son).squeeze()

            out_son = out_son.mean()
            iter_loss_son.append(out_son.item())

        if (epoch<epochs_only_sun):
            out = out_sun
        elif epoch<epochs_only_son:
            out = weight_son * out_son + out_h * weight_h
        else:
            out = out_sun + weight_son * out_son
        if not (done_with_son and done_with_sun):
            out.backward()
            optimizer.step()
            # make FC layer non-negative and with unit rows
            net.topnet.fc1.weight.data = projection_simplex_sort(net.topnet.fc1.weight.data)

        pbar.set_description('Epoch ' + str(epoch) + '. Loss SUN: ' + '%.4f' % (np.mean(iter_loss_sun))
                             + '. Loss SoN 0: ' + '%.4f' % (np.mean(iter_loss_son_level[0]))
                             + '. Loss SoN 2: ' + '%.4f' % (np.mean(iter_loss_son_level[np.minimum(2,len(iter_loss_son_level)-1)]))
                             + '. Loss SoN dense: ' + '%.4f' % (np.mean(iter_loss_son_level[-1]))
                             + '. Loss SoN avg: ' + '%.4f' % (np.mean(iter_loss_son_mean))
                             + '. Loss H: ' + '%.4f' % (np.mean(iter_loss_h)))

        pbar.update()
    pbar.close()
    epoch_loss_sun.append(np.mean(iter_loss_sun))
    epoch_loss_son.append(np.mean(iter_loss_son))


    torch.save(net.state_dict(), os.path.join(net_folder, out_net_name))

    ############################################################################################
    # VAL
    ############################################################################################
    net.eval()
    torch.cuda.empty_cache()
    iter_loss_sun = []
    iter_loss_son = []
    iter_loss_son_mean = []
    iter_loss_son_level = [[] for i in range(9)]
    dataloader_sun_val_for_epoch = iter(dataloader_sun_val)
    dataloader_son_val_for_epoch = iter(dataloader_son_val)
    done_with_sun = False
    done_with_son = False
    pbar = tqdm(total=iter_num_val)
    with torch.no_grad():
        for i in range(iter_num_val):
            # SUN Attributes
            if not done_with_sun:
                try:
                    data = dataloader_sun_val_for_epoch.next()
                except:
                    print('SUN exhausted in val')
                    done_with_sun = True
                    continue
                out = net(data['image'].to(device))
                out_sun = out['x_sun']
                maps = out['maps']
                out_sun = loss_sun(nl(out_sun), (data['label'].to(device) > 0).float())
                out_sun = out_sun.mean()
                iter_loss_sun.append(out_sun.item())

            # SoN
            if not done_with_son:
                try:
                    data = dataloader_son_val_for_epoch.next()
                except:
                    print('SoN exhausted in val')
                    done_with_son = True
                    continue
                out = net(data['image'].to(device))
                out_son = out['x_son']
                maps = out['maps']
                out_son_mean = []
                for k in range(len(out_son)):
                    out_son_mean.append(out_son[k])
                    out_son[k] = loss_son(out_son[k], data['label'].to(device))
                    iter_loss_son_level[k].append(out_son[k].mean().item())
                out_son = torch.stack(out_son)
                out_son_mean = torch.stack(out_son_mean)
                out_son_mean = out_son_mean[1:-1,:,:].mean(0)
                out_son_mean = loss_son(out_son_mean, data['label'].to(device))
                iter_loss_son_mean.append(out_son_mean.mean().item())

                out_son = out_son[-1,:,:].mean()
                iter_loss_son.append(out_son.item())
            pbar.set_description('Val loss SUN: ' + '%.4f' % (np.mean(iter_loss_sun))
                                 + '. Val loss SoN 0: ' + '%.4f' % (np.mean(iter_loss_son_level[0]))
                                 + '. Val loss SoN 2: ' + '%.4f' % (np.mean(iter_loss_son_level[2]))
                                 + '. Val loss SoN avg: ' + '%.4f' % (np.mean(iter_loss_son_mean)))
            pbar.update()

    pbar.close()
    epoch_loss_val_sun.append(np.mean(iter_loss_sun))
    epoch_loss_val_son.append(np.mean(iter_loss_son))

