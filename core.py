# encoding: utf-8

import torch.nn as nn
import torch
import numpy as np
# from utils import extract_weights,load_weights

def periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of x: (batch_size, Dp, N)
        return torch.cat((x, x[:,:,0:kernel_size-1]), -1)
    else:
        # shape of real x: (batch_size, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,0:kernel_size[0]-1,:]), -2)
        x = torch.cat((x, x[:,:,:,0:kernel_size[1]-1]), -1)
        return x

def complex_periodic_padding(x, kernel_size, dimensions):
    if dimensions == '1d':
        # shape of x: (batch_size, Dp, N)
        # shape of complex x: (batch_size, 2, Dp, N)
        return torch.cat((x, x[:,:,:,0:kernel_size-1]), -1)
    else:
        # shape of real x: (batch_size, Dp, Length, Width) 
        # shape of complex x: (batch_size, 2, Dp, Length, Width) 
        x = torch.cat((x, x[:,:,:,0:kernel_size[0]-1,:]), -2)
        x = torch.cat((x, x[:,:,:,:,0:kernel_size[1]-1]), -1)
        return x

def get_paras_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def gradient(y, x, grad_outputs=None):
    """compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

#--------------------------------------------------------------------
class CNN1d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',act=nn.ReLU,pbc=True):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN1d_layer,self).__init__()
        self.K = K 
        self._pbc = pbc
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv1d(Dp,F,self.K,1,0),act())
        else:
            self.conv = nn.Sequential(nn.Conv1d(F,F,self.K,1,0),act())

    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions='1d')
        x = self.conv(x)
        return x

class OutPut1d_layer(nn.Module):
    def __init__(self,K,F,output_size, pbc=True):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut1d_layer,self).__init__()
        self.K = K
        self._pbc = pbc
        self.linear = nn.Linear(F,output_size, bias=False)
    
    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions='1d')
        x = self.linear(x.sum(2))
        return x

# CNN 2D
#--------------------------------------------------------------------
class CNN2d_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',act=nn.ReLU, pbc=True):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN2d_layer,self).__init__()
        self.K = [K,K]
        self._pbc = pbc
        if layer_name == '1st':
            self.conv = nn.Sequential(nn.Conv2d(Dp,F,self.K,[1,1],0),act())
        else:
            self.conv = nn.Sequential(nn.Conv2d(F,F,self.K,[1,1],0),act())
        
    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions='2d')
        x = self.conv(x)
        return x

class OutPut2d_layer(nn.Module):
    def __init__(self,K,F,output_size,output_activation=False,pbc=True):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut2d_layer,self).__init__()
        self.K = [K,K]
        self._pbc=pbc
        self.linear = nn.Sequential(nn.Linear(F,output_size, bias=False))
        self.output_activation = output_activation
        self.output_size = output_size
    
    def forward(self,x):
        if self._pbc:
            x = periodic_padding(x, self.K, dimensions='2d')
        x = self.linear(x.sum(dim=[2,3]))
        if self.output_activation and self.output_size > 1:
            x[:,0] = torch.log(torch.sigmoid(x[:,0]))
            return x
        elif self.output_activation:
            return 2*np.pi*torch.sigmoid(x)
        else:
            return x

#--------------------------------------------------------------------
def mlp_cnn(state_size, K, F, layers=2, output_size=1, 
            output_activation=False, act=nn.ReLU, pbc=True):
    dimensions = len(state_size) - 1
    if dimensions == 1:
        """
        size of the input state (PBC): (batch size, Dp, N)
        N: length of the 1d lattice
        """
        Dp = state_size[-1]

        input_layer = CNN1d_layer(Dp, K, F, layer_name='1st', act=act, pbc=pbc)
        output_layer = OutPut1d_layer(K,F,output_size,pbc=pbc)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN1d_layer(Dp, K, F, layer_name='mid', act=act, pbc=pbc) for _ in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

        model = nn.Sequential(*cnn_layers)
    else:
        """
        size of the input state (PBC): (batch size, Dp, Length, Width)
        state size: (Length, Width, Dp)
        """
    
        Dp = state_size[-1]

        input_layer = CNN2d_layer(Dp, K, F, layer_name='1st', act=act, pbc=pbc)
        output_layer = OutPut2d_layer(K,F,output_size,output_activation,pbc=pbc)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN2d_layer(Dp, K, F, layer_name='mid', act=act, pbc=pbc) for _ in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

        model = nn.Sequential(*cnn_layers)

    model.apply(weight_init)
    return model


# COMPLEX NEURAL NETWORK
# ----------------------------------------------------------------
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, 
                    padding=0, dilation=1, groups=1, bias=True, dimensions='1d'):
        super(ComplexConv,self).__init__()

        ## Model components
        if dimensions == '1d':
            self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
            self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, 
                                padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
        return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()

        self.linear_re = nn.Linear(in_features, out_features, bias=bias)
        self.linear_im = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,in_features]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imaginary = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        output = torch.stack((real, imaginary),dim=1)
        return output

class ComplexLnCosh(nn.Module):
    def __init__(self):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexLnCosh, self).__init__()

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        z = torch.log(torch.cosh(z))
        return torch.stack((z.real, z.imag), dim=1)

class ComplexReLU(nn.Module):
    def __init__(self, relu_type='zReLU'):
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        super(ComplexReLU, self).__init__()
        self.type = relu_type
        self._bias = -1

    def forward(self, x):
        real = x[:,0]
        imag = x[:,1]
        z = real + 1j*imag
        if self.type == 'zReLU':
            mask = ((0 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'sReLU':
            mask = ((-np.pi/2 < z.angle()) * (z.angle() < np.pi/2)).float()
            return torch.stack((real*mask, imag*mask), dim=1)
        elif self.type == 'modReLU':
            z = torch.relu(z.abs() + self._bias) * torch.exp(1j * z.angle()) 
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus':
            z = torch.log(1. + torch.exp(z))
            return torch.stack((z.real, z.imag), dim=1)
        elif self.type == 'softplus2':
            z = torch.log(1./2. + torch.exp(z)/2.)
            return torch.stack((z.real, z.imag), dim=1)
        else:
            return torch.stack((torch.relu(real), torch.relu(imag)), dim=1)

# CNN 1D 
#--------------------------------------------------------------------
class CNN1d_complex_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',relu_type='sReLU',pbc=True, bias=True):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN1d_complex_layer,self).__init__()
        self.K = K 
        self._pbc = pbc
        self.layer_name = layer_name
        complex_act = ComplexReLU(relu_type)
        if layer_name == '1st':
            complex_conv = ComplexConv(Dp,F,self.K,1,0, dimensions='1d', bias=bias)
        else:
            complex_conv = ComplexConv(F,F,self.K,1,0, dimensions='1d', bias=bias)
        self.conv = nn.Sequential(*[complex_conv, complex_act])

    def forward(self, x):
        if self.layer_name == '1st':
            x = torch.stack((x, torch.zeros_like(x)), dim=1)
        if self._pbc:
            x = complex_periodic_padding(x, self.K, dimensions='1d')
        x = self.conv(x)
        return x

class OutPut1d_complex_layer(nn.Module):
    def __init__(self,K,F,pbc=True):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut1d_complex_layer,self).__init__()
        self.K = K
        self._pbc=pbc
        self.linear = ComplexLinear(F,1, bias=False)
    
    def forward(self,x):
        if self._pbc:
            x = complex_periodic_padding(x, self.K, dimensions='1d')
        # shape of complex x: (batch_size, 2, F, N)
        x = self.linear(x.sum(-1)) 
        return x.squeeze(-1)

# CNN 2D 
#--------------------------------------------------------------------
class CNN2d_complex_layer(nn.Module):
    def __init__(self,Dp,K,F,layer_name='mid',relu_type='sReLU', pbc=True, bias=True):
        """
        Dp = 1: value encoding
        Dp > 1: onehot encoding
        """
        super(CNN2d_complex_layer,self).__init__()
        self.K = [K,K]
        self._pbc = pbc
        self.layer_name = layer_name
        complex_act = ComplexReLU(relu_type)
        # complex_act = ComplexLnCosh()
        if layer_name == '1st':
            complex_conv = ComplexConv(Dp,F,self.K,1,0, dimensions='2d', bias=bias)
        else:
            complex_conv = ComplexConv(F,F,self.K,1,0, dimensions='2d', bias=bias)
        self.conv = nn.Sequential(*[complex_conv, complex_act])
        
    def forward(self,x):
        if self.layer_name == '1st':
            x = torch.stack((x, torch.zeros_like(x)), dim=1)
        if self._pbc:
            x = complex_periodic_padding(x, self.K, dimensions='2d')
        x = self.conv(x)
        return x

class OutPut2d_complex_layer(nn.Module):
    def __init__(self,K,F,pbc=True):
        """
        output size = 1: logphi
        output size = 2: logphi, theta
        """
        super(OutPut2d_complex_layer,self).__init__()
        self.K = [K,K]
        self._pbc = pbc
        self.linear = ComplexLinear(F,1,bias=False)
    
    def forward(self,x):
        if self._pbc:
            x = complex_periodic_padding(x, self.K, dimensions='2d')
        # shape of complex x: (batch_size, 2, F, L, W)
        return self.linear(x.sum(dim=[3,4])).squeeze(-1)
#--------------------------------------------------------------------
def mlp_cnn_complex(state_size, K, F, layers=2, relu_type='sReLU', pbc=True, bias=True):
    dimensions = len(state_size) - 1
    if dimensions == 1:
        """
        size of the input state (PBC): (batch size, Dp, N)
        N: length of the 1d lattice
        """
        Dp = state_size[-1]

        input_layer = CNN1d_complex_layer(Dp, K, F, layer_name='1st', relu_type=relu_type, 
                                            pbc=pbc, bias=bias)
        output_layer = OutPut1d_complex_layer(K,F, pbc=pbc)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN1d_complex_layer(Dp, K, F, layer_name='mid', relu_type=relu_type, pbc=pbc, bias=bias) for _ in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if bias:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        model = nn.Sequential(*cnn_layers)
    else:
        """
        size of the input state (PBC): (batch size, Dp, Length, Width)
        state size: (Length, Width, Dp)
        """
    
        Dp = state_size[-1]

        input_layer = CNN2d_complex_layer(Dp, K, F, layer_name='1st', relu_type=relu_type, pbc=pbc, bias=bias)
        output_layer = OutPut2d_complex_layer(K,F, pbc=pbc)

        # input layer
        cnn_layers = [input_layer]
        cnn_layers += [CNN2d_complex_layer(Dp, K, F, layer_name='mid', relu_type=relu_type, pbc=pbc, bias=bias) for _ in range(1,layers)]
        cnn_layers += [output_layer]

        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if bias:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        model = nn.Sequential(*cnn_layers)

    model.apply(weight_init)
    return model

# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # logphi_model = CNNnet_1d(10,2)
    logphi_model = mlp_cnn_complex([10,10,2], 2, 2, layers=1, relu_type='sReLU', bias=True)
    op_model = mlp_cnn_complex([10,10,2], 2, 2, layers=1, relu_type='sReLU', bias=True)
    print(logphi_model)
    print(get_paras_number(logphi_model))

    from operators.tfim_spin2d import get_init_state
    state0 = get_init_state([10,10,2], kind='rand', n_size=1)
    print(state0.shape) 

    phi = logphi_model(torch.from_numpy(state0).float())
    logphi = phi[:,0].reshape(1,-1)
    theta = phi[:,1].reshape(1,-1)
    print(phi.shape)
    print(logphi)

    op_model.load_state_dict(logphi_model.state_dict())
    phi2 = op_model(torch.from_numpy(state0).float())
    print(logphi - phi2[:,0].reshape(1,-1))
    

    # logphi = phi[:,0]
    # logphi.backward()
    logphi_model.zero_grad()
    logphi.sum().backward(retain_graph=True)
    logphi_model.zero_grad()
    theta.sum().backward(retain_graph=True)
    for name,p in logphi_model.named_parameters():
        print(name.split(".")[3], p.numel())
        grads = gradient(theta.sum(), p)
        print(grads)
        print(p.grad)

    state0s = np.repeat(state0, 10, axis=0)
    print(state0s.shape)
    print(state0s[0] - state0s[1])
    
    '''
    from utils import extract_weights, load_weights
    import copy
    ps = logphi_model.state_dict()
    op_psi_model = copy.deepcopy(logphi_model)
    params, names = extract_weights(op_psi_model)
    name_num = []
    for name in names:
        name_num.append(int(name.split(".")[0]))
    print(name_num)
    name_num, counts = np.unique(name_num, return_counts=True, axis=0)
    print(counts)
    length = int(names[-1].split(".")[0])
    print(ps[names[-1]])
    
    def forward(*new_param):
        load_weights(op_psi_model, names, new_param)
        out = op_psi_model(torch.from_numpy(state0).float())
        return out
    
    cnt = 0
    dydws = torch.autograd.functional.jacobian(forward, params)
    for net_layer in counts:
        step = net_layer // 2
        for i in range(step):
            index = i + cnt
            dres = dydws[index].reshape(10,2,-1)
            dims = dydws[index + step].reshape(10,2,-1)
            Oks = 0.5*(dres[:,0,:] + dims[:,1,:]) - 0.5*1j*(dims[:,0,:] - dres[:,1,:])
            Oks_conj = Oks.conj()
            print(names[index])
            print(names[index+step])
        cnt += net_layer
    
    logphi_model.load_state_dict(ps)
    
    # print(params.shape)
    # nn.utils.vector_to_parameters(params, logphi_model.parameters())
    #print(phi.norm(dim=1,keepdim=True))
    '''
    '''
    model2, params = get_resnet18()
    print(params[0].shape)
    
    print(model2(params))
    
    '''
    # y = jacobian(model2, params[0])
    # print(y.shape)