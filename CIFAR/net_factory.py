import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from gcn.modules import GConv 
from utils import accuracy, AverageMeter, save_checkpoint, visualize_graph, get_parameters_size


def conv3x3(in_planes, out_planes, stride=1, nChannel=4, nScale=1):
    return GConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, M = nChannel, nScale=nScale, expand=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride, nChannel, nScale):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes * nChannel)
        self.conv1 = GConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, M=nChannel, nScale=nScale)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes * nChannel)
        self.conv2 = GConv(planes, planes, kernel_size=3, padding=1, bias=False, M=nChannel, nScale=nScale)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                GConv(in_planes, planes, kernel_size=1, stride=stride, bias=False, M=nChannel, nScale=nScale),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_Gabor_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, nChannel=4, nScale=1):
        super(Wide_Gabor_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Gabor-Resnet %dx%dx%dx%d' %(depth, k, nChannel, nScale))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0],nChannel=nChannel, nScale=nScale)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, nChannel=nChannel, nScale=nScale)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, nChannel=nChannel, nScale=nScale)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, nChannel=nChannel, nScale=nScale)
        self.bn1 = nn.BatchNorm2d(nStages[3] * nChannel, momentum=0.9)
        self.linear = nn.Linear(nStages[3] * nChannel, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, nChannel, nScale):
        strides = [stride] + [1,]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, nChannel, nScale))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def get_network_fn(name, depth, widen_factor, dropout_rate, num_classes, nChannel, nScale):
    networks_zoo = {
    'gcn': Wide_Gabor_ResNet(depth, widen_factor, dropout_rate, num_classes, nChannel=nChannel, nScale=nScale),
    }
    if name is '':
        raise ValueError('Specify the network to train. All networks available:{}'.format(networks_zoo.keys()))
    elif name not in networks_zoo:
        raise ValueError('Name of network unknown {}. All networks available:{}'.format(name, networks_zoo.keys()))
    return networks_zoo[name]

def test():
    a = Variable(torch.randn(2,3,32,32))
    model = get_network_fn('gcn', 40, 2, 0.5, 10, nChannel=4, nScale=1)
    print model
    b = model(a)
    print b.size()
    print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))



if __name__ == '__main__':
    test()