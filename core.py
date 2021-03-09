# encoding: utf-8

import torch.nn as nn
import torch

def periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of x: (batch_size, Dp, N)
        return torch.cat((x, x[:,:,0:kernel_size-1]), 2)
    else:
        # shape of x: (batch_size, Dp, Length, Width)
        x = torch.cat((x, x[:,:,0:kernel_size[0]-1,:]), 2)
        x = torch.cat((x, x[:,:,:,0:kernel_size[1]-1]), 3)
        return x

def get_paras_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#----------------------------------------------------------------
class CNNnet_1d(nn.Module):
    """
    size of the input state (PBC): (batch size, Dp, N)
    """
    def __init__(self,Dp,K,F):
        super(CNNnet_1d, self).__init__()
        self.K = K
        self.F = F
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=Dp,
                out_channels=self.F,
                kernel_size=self.K,
                stride=1,
                padding=0
            ),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.F,self.F,self.K,1,0),
            nn.ReLU()
            )
        self.linear = nn.Linear(self.F,1)

    def forward(self, x):
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.conv1(x)
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.conv2(x)
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.linear(x.sum(2))
        return x

#--------------------------------------------------------------------
class CNN1d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN1d_layer,self).__init__()
        self.K = K 
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv1d(Dp,F,self.K,1,0),nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv1d(F,F,self.K,1,0),nn.ReLU())

    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.conv(x)
        return x

class OutPut1d_layer(nn.Module):
    def __init__(self,K,F,output_size):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut1d_layer,self).__init__()
        self.K = K
        self.linear = nn.Linear(F,output_size)
    
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.linear(x.sum(2))
        return x

#--------------------------------------------------------------------
class CNN2d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid'):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN2d_layer,self).__init__()
        self.K = [K,K]
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv2d(Dp,F,self.K,[1,1],0),nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(F,F,self.K,[1,1],0),nn.ReLU())
        
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='2d')
        x = self.conv(x)
        return x

class OutPut2d_layer(nn.Module):
    def __init__(self,K,F,output_size):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut2d_layer,self).__init__()
        self.K = [K,K]
        self.linear = nn.Linear(F,output_size)
    
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='2d')
        x = self.linear(x.sum(2).sum(2))
        return x

#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F, layers=2, output_size=1):
    dimensions = len(state_size) - 1
    if dimensions == 1:
        """
        size of the input state (PBC): (batch size, Dp, N)
        N: length of the 1d lattice
        """
        Dp = state_size[-1]

        input_layer = CNN1d_layer(Dp, K, F, layer_name='1st')
        hid_layer = CNN1d_layer(Dp, K, F, layer_name='mid')
        output_layer = OutPut1d_layer(K,F,output_size)

        # input layer
        cnn_layers = [input_layer]
        for i in range(1,layers):
            cnn_layers += [hid_layer]
        
        cnn_layers += [output_layer]

        return nn.Sequential(*cnn_layers)
    else:
        """
        size of the input state (PBC): (batch size, Dp, Length, Width)
        state size: (Length, Width, Dp)
        """
        Dp = state_size[-1]
        input_layer = CNN2d_layer(Dp, K, F, layer_name='1st')
        hid_layer = CNN2d_layer(Dp, K, F, layer_name='mid')
        output_layer = OutPut2d_layer(K,F,output_size)

        # input layer
        cnn_layers = [input_layer]
        for i in range(1,layers):
            cnn_layers += [hid_layer]
        
        cnn_layers += [output_layer]

        return nn.Sequential(*cnn_layers)

if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    logphi_model = mlp_cnn([10, 10, 2], 3, 4, output_size=2)
    print(logphi_model)
    print(get_paras_number(logphi_model))

    from tfim_spin2d import get_init_state
    state0 = get_init_state([10, 10, 2], kind='rand', n_size=10)
    print(state0.shape) 

    phi = logphi_model(torch.from_numpy(state0).float())
    print(phi)
    print(phi[:,0])
    print(phi[:,1])


   