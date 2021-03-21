# encoding: utf-8

import torch.nn as nn
import torch
import numpy as np
# from utils import extract_weights,load_weights

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

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

#--------------------------------------------------------------------
class CNN1d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',act=nn.ReLU):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN1d_layer,self).__init__()
        self.K = K 
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv1d(Dp,F,self.K,1,0),act())
        else:
            self.conv = nn.Sequential(nn.Conv1d(F,F,self.K,1,0),act())

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
        self.linear = nn.Linear(F,output_size, bias=False)
    
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='1d')
        x = self.linear(x.sum(2))
        return x

#--------------------------------------------------------------------
class CNN2d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',act=nn.ReLU):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN2d_layer,self).__init__()
        self.K = [K,K]
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv2d(Dp,F,self.K,[1,1],0),act())
        else:
            self.conv = nn.Sequential(nn.Conv2d(F,F,self.K,[1,1],0),act())
        
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='2d')
        x = self.conv(x)
        return x

class OutPut2d_layer(nn.Module):
    def __init__(self,K,F,output_size,output_activation=False):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut2d_layer,self).__init__()
        self.K = [K,K]
        self.linear = nn.Sequential(nn.Linear(F,output_size, bias=False))
        self.output_activation = output_activation
        self.output_size = output_size
    
    def forward(self,x):
        x = periodic_padding(x, self.K, dimensions='2d')
        x = self.linear(x.sum(2).sum(2))
        if self.output_activation and self.output_size > 1:
            x[:,0] = torch.log(torch.sigmoid(x[:,0]))
            return x
        elif self.output_activation:
            return torch.log(torch.sigmoid(x))
        else:
            return x

#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F, layers=2, output_size=1, output_activation=False, act=nn.ReLU):
    dimensions = len(state_size) - 1
    if dimensions == 1:
        """
        size of the input state (PBC): (batch size, Dp, N)
        N: length of the 1d lattice
        """
        Dp = state_size[-1]

        input_layer = CNN1d_layer(Dp, K, F, layer_name='1st', act=act)
        output_layer = OutPut1d_layer(K,F,output_size)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN1d_layer(Dp, K, F, layer_name='mid', act=act) for _ in range(1,layers)]
        cnn_layers += [output_layer]

        return nn.Sequential(*cnn_layers)
    else:
        """
        size of the input state (PBC): (batch size, Dp, Length, Width)
        state size: (Length, Width, Dp)
        """
    
        Dp = state_size[-1]
        input_layer = CNN2d_layer(Dp, K, F, layer_name='1st', act=act)
        output_layer = OutPut2d_layer(K,F,output_size,output_activation)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN2d_layer(Dp, K, F, layer_name='mid', act=act) for _ in range(1,layers)]
        cnn_layers += [output_layer]
        return nn.Sequential(*cnn_layers)

if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    logphi_model = mlp_cnn([10, 10, 2], 3, 4, layers=4, output_size=2, act=nn.Softplus)
    op_model = mlp_cnn([10, 10, 2], 3, 4, layers=4, output_size=2, act=nn.Softplus)
    print(logphi_model)
    print(get_paras_number(logphi_model))

    from tfim_spin2d import get_init_state
    state0 = get_init_state([10, 10, 2], kind='rand', n_size=10)
    print(state0.shape) 

    phi = logphi_model(torch.from_numpy(state0).float())
    print(phi)
    params = nn.utils.parameters_to_vector(logphi_model.parameters())

    # logphi = phi[:,0]
    # logphi.backward()
    # for p in logphi_model.parameters():

    
    # print(params.shape)
    # nn.utils.vector_to_parameters(params, logphi_model.parameters())
    #print(phi.norm(dim=1,keepdim=True))
    from utils import extract_weights, load_weights, _del_nested_attr, _set_nested_attr
    from torch.autograd.functional import jacobian
    import time
    import copy

    # op_model.load_state_dict(logphi_model.state_dict())
    op_model = copy.deepcopy(logphi_model)
    params, names = extract_weights(op_model)

    def forward(*new_param):
        load_weights(op_model, names, new_param)
        out = op_model(torch.from_numpy(state0).float())
        return out
    
    tic = time.time()
    y = jacobian(forward, params)
    print(time.time() - tic)
    print(y[0])

    t = 0
    op_model = copy.deepcopy(logphi_model)
    cnt = 0
    for name, p in list(logphi_model.named_parameters()):
        _del_nested_attr(op_model, name.split("."))
        para = p.detach().requires_grad_()
        # print(para.shape)

        def forward(new_param):
            _set_nested_attr(op_model, name.split("."), new_param)
            out = op_model(torch.from_numpy(state0).float())
            return out

        tic = time.time()
        y = jacobian(forward, para, create_graph=True)
        t += time.time()-tic
        # y = y.reshape(10,2,-1)
        if cnt == 0:
            print(y)
        cnt += 1
        # _set_nested_attr(logphi_model, name.split("."), para)
    print(t)
    # print(logphi_model.state_dict())
    
    '''
    model2, params = get_resnet18()
    print(params[0].shape)
    
    print(model2(params))
    
    '''
    # y = jacobian(model2, params[0])
    # print(y.shape)