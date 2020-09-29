from skimage import io, transform
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt


class SoNDataset(Dataset):
    """SoN scenicness dataset."""

    def __init__(self, im_paths,labels,locations = None, tr=None, hist = None):
        self.im_paths = im_paths
        self.labels = labels
        self.hist = hist
        self.label_avg = np.float32(labels.mean(axis=0))
        self.locations = locations
        self.transform = tr

    def __len__(self):
        return len(self.im_paths)


    def __getitem__(self, idx):
        img_name = self.im_paths[idx]
        try:
            image = io.imread(img_name)
        except:
            return None
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        if self.locations is not None:
            sample['loc'] = self.locations[idx, :]
        if self.hist is not None:
            hist = self.hist[idx, :]
            sample['hist'] = hist
        return sample

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class SUNAttributesDataset(Dataset):
    """SUN Attributes dataset."""

    def __init__(self, data_path,im_names,labels,attr_names, tr=None):
        self.data_path = data_path
        self.im_names = im_names
        self.labels = labels
        self.attr_names = attr_names
        self.transform = tr

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, 'images',
                                self.im_names[idx])
        image = io.imread(img_name)
        s = image.shape
        if len(s) > 3:
            print(img_name)
        if len(s) == 2:
            image = image[:, :, np.newaxis][:, :, [0, 0, 0]]
        if image.shape[2] != 3:
            image = image[:,:,0:3]
        image = np.ascontiguousarray(image)
        label = np.float32(self.labels[idx,:])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.transpose(image,(2, 0, 1))
        image = image.astype('float32') / 255

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        done_resizing = False
        while not done_resizing:
            try:
                img = transform.resize(image, (new_h, new_w),mode='constant',anti_aliasing=True)
                done_resizing = True
            except:
                print('Issue resizing. Trying again.')


        return {'image': img, 'label': label}


class Rotate(object):
    """Rotate the image in a sample to a given size.

    Args:
        output_size (float): Maximum rotation.
    """

    def __init__(self, max_angle):
        assert isinstance(max_angle, (int, float))
        self.max_angle = max_angle

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        angle = (np.random.rand()-0.5)*2*self.max_angle
        done_rotating = False
        while not done_rotating:
            try:
                img = transform.rotate(image, angle)
                done_rotating = True
            except:
                print('Issue rotating. Trying again.')


        return {'image': img, 'label': label}

class VerticalFlip(object):
    """Flip the image in a sample with probability p.

    Args:
        p (float): Probability of vertical flip.
    """

    def __init__(self, p = 0.5):
        assert isinstance(p, (float))
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if np.random.rand() < self.p:
            image = np.fliplr(image)


        return {'image': image, 'label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}

def inspect_dataset(dataset,attr_names):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for idx in range(20, 2000, 20):
        print(idx)
        show_image(dataset.__getitem__(idx), attr_names, fig, axs)
        axs[0].cla()
        axs[1].cla()

def show_image(sample,names,fig,axs):
    axs[0].imshow(sample['image'])
    label = sample['label']
    names = [names[i] for i in np.where(label>0)[0]]
    label = label[label>0]
    axs[1].barh(np.arange(len(label)),label)
    axs[1].set_yticks(np.arange(len(label)))
    axs[1].set_yticklabels(names)
    fig.tight_layout()
    plt.pause(0.0001)
    plt.waitforbuttonpress(timeout=-1)


class NetSUNSoNTopBaseMaps(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSUNSoNTopBaseMaps, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 33, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)
        self.pool_sun = nn.AdaptiveAvgPool2d(1)
        self.pool_son = nn.AdaptiveAvgPool2d(1)

        self.conv_son.bias.data[0].fill_(label_avg[0])
        self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        maps_sun = F.relu(self.conv_sun(x))
        maps_son = self.conv_son(x)

        # pool to get attribute vector
        x_sun = (self.pool_sun(maps_sun)).view(-1, 33)
        x_son = (self.pool_son(maps_son)).view(-1, 2)


        return x_sun, x_son, maps_sun, maps_son

class NetSUNSoNTopBase(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSUNSoNTopBase, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 34, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        #self.conv_son.bias.data[0].fill_(label_avg[0])
        #self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        x_sun = self.pool(F.relu(self.conv_sun(x))).squeeze()
        x_son = self.pool(self.conv_son(x)).squeeze()


        return x_sun, x_son

class NetSoNTopBase(nn.Module):
    def __init__(self,label_avg=[0,0]):
        super(NetSoNTopBase, self).__init__()
        self.conv_sun = nn.Conv2d(2048, 3, 1)
        self.conv_son = nn.Conv2d(2048, 2, 1)
        self.pool_sun = nn.AdaptiveAvgPool2d(1)
        self.pool_son = nn.AdaptiveAvgPool2d(1)

        self.conv_son.bias.data[0].fill_(label_avg[0])
        self.conv_son.bias.data[1].fill_(label_avg[1])

    def forward(self, x):
        # get maps of attributes
        maps_sun = F.relu(self.conv_sun(x))
        maps_son = self.conv_son(x)

        # pool to get attribute vector
        x_sun = (self.pool_sun(maps_sun)).view(-1, 33)
        x_son = (self.pool_son(maps_son)).view(-1, 2)


        return x_sun, x_son, maps_sun, maps_son


class NetSUNTop(nn.Module):
    def __init__(self):
        super(NetSUNTop, self).__init__()
        self.conv1 = nn.Conv2d(2048, 102, 1)


    def forward(self, x):
        # get maps of attributes
        maps = self.conv1(x).relu()

        maps[:, :, 0:1, :] = 0
        maps[:, :, -1:, :] = 0
        maps[:, :, :, 0:1] = 0
        maps[:, :, :, -1:] = 0

        return maps


class DistanceLayer(nn.Linear):

    def __init__(self, init_loc):
        in_features = init_loc.shape[1]
        out_features = init_loc.shape[0]
        bias = True
        super(DistanceLayer, self).__init__(in_features, out_features, bias)
        self.weight.data = torch.Tensor(init_loc)
        self.bias.data.fill_(0)
        #self.sigma = torch.nn.Parameter(self.bias.data.clone())
        #self.sigma.data.fill_(0)

    def forward(self, input):

        sim = torch.nn.functional.linear(input,self.weight)
        dist = (input**2).sum(1,keepdim=True) + (self.weight**2).sum(1,keepdim=True).transpose(1,0) - 2 * sim
        dist = dist.relu()
        dist = dist - dist.min(1,keepdim=True)[0]
        #out = 1 / (dist.relu() * self.sigma.unsqueeze(0).exp() + self.bias.unsqueeze(0).relu() + 1e-5)
        #out = (-dist * self.bias.unsqueeze(0)).exp()
        out = 1 / ( dist + self.bias.unsqueeze(0) + 1e-2)
        #out = torch.softmax(-dist*(self.bias.unsqueeze(0)+0.1),1)
        out = out / out.sum(1,keepdim=True)

        return out

class AddScalar(nn.Module):
    def __init__(self,value):
        super(AddScalar, self).__init__()
        self.value = value
    def forward(self,x):
        return self.value + x

class FullNetGeo(nn.Module):
    def __init__(self,basenet,topnet):
        super(FullNetGeo, self).__init__()
        self.basenet = basenet
        self.topnet = topnet
    def forward(self,x):
        x = self.basenet(x)
        out = self.topnet(x)
        return out

class RouteFcMaxActDual(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(RouteFcMaxActDual, self).__init__(in_features, out_features, bias)
        self.weight_neg = torch.nn.Parameter(self.weight.clone())

    def forward(self, input, do_sparse = True, topk = 1):
        if do_sparse:
            vote = input[:, None, :] * self.weight - input[:, None, :] * self.weight_neg
            val = vote.abs().topk(topk, 2)
            out = torch.gather(vote, 2, val[1]).sum(2)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = torch.nn.functional.linear(input, self.weight, self.bias)
            out = out - torch.nn.functional.linear(input, self.weight_neg)
        return out

class RouteFcMaxActAbs(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(RouteFcMaxActAbs, self).__init__(in_features, out_features, bias)

    def forward(self, input, do_sparse = True, topk = 1):
        if do_sparse:
            input = input[:, None, :]
            vote = input * self.weight
            val = vote.abs().topk(topk, 2)
            out = torch.gather(vote, 2, val[1])
            #presence = torch.gather(input, 2, val[1])
            #presence = presence / (presence.sum(2,keepdim=True)+1e-3)
            #out = out * presence.detach()
            #out = out[:,:,:-1].detach().sum(2) + out[:,:,-1]
            out = out.sum(2)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = torch.nn.functional.linear(input, self.weight, self.bias)
        return out

class RouteFcMaxAct(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(RouteFcMaxAct, self).__init__(in_features, out_features, bias)

    def forward(self, input, do_sparse = True, topk = 1):
        if do_sparse:
            vote = input[:, None, :] * self.weight
            out = vote.topk(topk, 2)[0].sum(2)
            #out = torch.gather(vote, 2, val[1]).sum(2)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = torch.nn.functional.linear(input, self.weight, self.bias)
        return out

class ThresholdActivations(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ThresholdActivations, self).__init__(in_features, out_features, bias)

    def forward(self,input,thr=0):
        input = input * (input > thr)
        out = torch.nn.functional.linear(input, self.weight, self.bias)
        return out

class TopKActivations(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(TopKActivations, self).__init__(in_features, out_features, bias)

    def forward(self,input,topk=0):
        sorted_input = torch.sort(input, descending=True)[0]
        thr = sorted_input[:,topk-1:topk]
        input = input * (input >= thr)
        out = torch.nn.functional.linear(input, self.weight, self.bias)
        return out


class ThresholdContributions(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ThresholdContributions, self).__init__(in_features, out_features, bias)

    def forward(self, input, thr=0):
        contrib = input[:, None, :] * self.weight
        contrib = contrib * (contrib.abs() >= thr)
        out = contrib.sum(2)
        if self.bias is not None:
            out = out + self.bias
        return out

def projection_simplex_sort(v, z=1):

    n_features = v.size(1)
    u,_ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u,1) - z
    ind = torch.arange(n_features).type_as(v) + 1
    cond = u - cssv / ind > 0
    #rho = ind[cond][-1]
    rho,ind_rho = (ind*cond).max(1)
    #theta = cssv[cond][-1] / float(rho)
    theta = torch.gather(cssv,1,ind_rho[:,None]) / rho[:,None]
    w = torch.clamp(v - theta, min=0)
    return w

class NetSoNTopSINReg(nn.Module):
    def __init__(self,label_avg,n_attributes=102,n_groups = 200, do_negations=False, topks=[1,2, 3, 4, 5, 6, 7, 8]):
        super(NetSoNTopSINReg, self).__init__()

        self.do_negations = do_negations
        self.topks = topks
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(n_attributes, n_groups, bias = False)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = RouteFcMaxActAbs(n_groups, 1, bias = False)
        #self.fc2 = TopKActivations(100, 1, bias=False)
        #self.fc2 = ThresholdActivations(100,1,bias=False)
        #self.fc2 = ThresholdContributions(100, 1, bias=False)

        #self.fc1.weight.data.fill_(0)
        #while (self.fc1.weight.data.sum(1)==0).sum() > 0:
        #self.fc1.weight.data.bernoulli_(1)

        #self.fc1.weight.data = self.fc1.weight.data - self.fc1.weight.data.max(1, keepdim=True)[0]+1e-3
        #self.fc1.weight.data.clamp_min_(0)
        #self.fc1.weight.data = self.fc1.weight.data / (self.fc1.weight.data.max(1, keepdim=True)[0]+1e-3) + 5e-2
        #self.fc1.weight.data = self.fc1.weight.data.abs()
        self.fc1.weight.data = self.fc1.weight.data.relu() / (
            self.fc1.weight.data.relu().sum(1, keepdim=True))
        #self.fc2.weight.data[:] = (torch.arange(-0.5,0.5,1/n_groups))
        #self.fc2.weight.data = self.fc2.weight.data + self.fc2.weight.data.sign()*4
        #self.fc2.weight.data = self.fc2.weight.data.sort()[0]
        self.fc2.weight.data.fill_(0)
        #self.fc1.weight.data = self.fc1.weight.data - self.fc1.weight.data.std()/1
        self.avg_value = label_avg[0]

        # self.fc_level = nn.Sequential(*[
        #     nn.Linear(200,150),
        #     nn.BatchNorm1d(150),
        #     nn.ReLU(),
        #     nn.Linear(150,150),
        #     nn.BatchNorm1d(150),
        #     nn.ReLU(),
        #     nn.Linear(150, 9)
        # ])
        # self.fc_level[-1].weight.data.fill_(0)
        #self.fc_level.fc1 = nn.Linear(20,50)
        #self.fc_level.fc2 = nn.Linear(50,5)
        #self.fc_level.fc2.weight.data.fill_(0)
        #list(self.fc_level)[-1].weight.data.fill_(0)

    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool(maps)).squeeze()



        b = 1e-8

        if self.do_negations:
            x_sun_prob = torch.cat([(x_sun.relu()+b).tanh(),1-x_sun.relu().tanh()*(1-b)],1)
        else:
            x_sun_prob = (x_sun.relu()+b).tanh()

        #x_groups = self.fc1(x_sun.tanh()).relu()**2
        x_sun_log = x_sun_prob.log()
        x_groups_log = self.fc1(x_sun_log)
        #group_sum = self.fc1.weight.data.relu().detach().sum(1, keepdim=True).transpose(1,0)
        #x_groups = (x_groups_log-np.log(2)*group_sum).exp()-1e-4
        #x_groups = x_groups.relu()
        x_groups = (x_groups_log).exp() - b

        #x_level = self.fc_level(x_groups*self.fc2.weight)

        #x_groups = x_groups.bernoulli()

        #x_groups = self.dropout(x_groups)
        x = []

        for topk in self.topks:
            x.append(self.fc2(x_groups,topk=topk, do_sparse=True) + self.avg_value)
        x.append(self.fc2(x_groups, topk=topk, do_sparse=False) + self.avg_value)

        #for thr in np.arange(0,0.6,0.12):
        #    x.append(self.fc2(x_groups, thr=thr) + self.avg_value)

        #for thr in [0.6,0.5,0.4,0.3,0.2,0.1,0,0,0]:
        #    x.append(self.fc2(x_groups, thr=thr) + self.avg_value)

        x_son = x




        out = {'x_sun': x_sun, 'x_son': x_son, 'maps': maps , 'x_groups_log': x_groups_log }

        return out

class NetSoNTopSIAMReg(nn.Module):
    def __init__(self,label_avg,n_attributes=102, topks=[1,2, 3, 4, 5, 6, 7, 8]):
        super(NetSoNTopSIAMReg, self).__init__()

        self.topks = topks
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Just a placeholder
        self.fc1 = RouteFcMaxActAbs(1, 1, bias=False)

        self.fc2 = RouteFcMaxActAbs(n_attributes, 1, bias = False)



        self.fc2.weight.data.fill_(0)
        self.avg_value = label_avg[0]


    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool(maps)).squeeze()

        x = []

        for topk in self.topks:
            x.append(self.fc2(x_sun,topk=topk, do_sparse=True) + self.avg_value)
        x.append(self.fc2(x_sun, topk=topk, do_sparse=False) + self.avg_value)
        x_son = x
        out = {'x_sun': x_sun, 'x_son': x_son, 'maps': maps , 'x_groups_log': x_sun}

        return out

class NetSoNTopSIN(nn.Module):
    def __init__(self,label_avg):
        super(NetSoNTopSIN, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(33, 100, bias = False)
        self.fc2 = RouteFcMaxAct(100, 10, bias = False)
        #self.fc2.bias.data.fill_(label_avg[0])
        #self.fc2.weight.data.bernoulli_()
        #self.fc2.weight.data = self.fc2.weight.data * 0.01
        #self.fc2.weight_neg.data.fill_(1)
        #self.fc2.weight.data.fill_(0)
        self.avg_value = label_avg[0]

    def forward(self, maps):

        # pool to get attribute vector
        x_sun = (self.pool(maps[:,0:33,:,:])).view(-1, 33)

        x_groups = self.fc1(x_sun.tanh()).relu()
        x = []
        for topk in [3,4,5,6,7,10,15,20]:
            x.append(self.fc2(x_groups,topk=topk))
        x.append(self.fc2(x_groups, do_sparse=False) )
        x_son = x
        #x_son = torch.stack(x).mean(0)
        #x = x / (x.sum(1,keepdim=True)+1e-5)
        #x_son = x * (torch.arange(start=1, end=10, step=1).type_as(x).unsqueeze(0))
        #x_son = x_son.sum(1)

        out = {'x_sun': x_sun, 'x_son': x_son, 'maps': maps}

        return out

def setup_model(init_net_name=None, init_net_folder = '.', regression_avg = [0],n_attributes=102,n_groups=200,do_negations=False,topks=[1,2,3,4,5,6,7,8]):
    # Prepare initial model
    basenet = models.resnet50(pretrained=True)
    if n_groups > 0:
        topnet = NetSoNTopSINReg(regression_avg,n_attributes=n_attributes,n_groups=n_groups,do_negations=do_negations,topks=topks)
    else:
        topnet = NetSoNTopSIAMReg(regression_avg,n_attributes=n_attributes,topks=[n_attributes])

    if init_net_name is not None:
        # If the initial model has only been trained on SUN Attributes
        if init_net_name[0:8] == 'sun_base':
            net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop())
            net.load_state_dict(torch.load(os.path.join(init_net_folder,init_net_name),map_location='cpu'))
            #net = nn.Sequential(*list(net.children()), NetSoNTopGeo(init_loc,dataset_son_train.label_avg)).to(device)
            net = FullNetGeo(net,topnet)
        # If it has been trained both on SUN and SoN
        else:
            net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop())
            net = FullNetGeo(net, topnet)
            net.load_state_dict(torch.load(os.path.join(init_net_folder,init_net_name),map_location='cpu'),strict=False)
    else:
        # Initialize random model
        net = nn.Sequential(*list(basenet.children())[:-2], NetSUNTop())
        net = FullNetGeo(net, topnet)
    return  net

