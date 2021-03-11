# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version 2.0 updates: compatible with 2d system, replay buffer
# version 3.0: double CNN (real and imag), imaginary time propagation

import numpy as np
import torch
import torch.nn as nn
from mcmc_sampler_complexv2 import MCsampler
from core import mlp_cnn, get_paras_number, gradient
from utils import get_logger, _get_unique_states
import time
import os

# gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.device("cpu")
cpu = torch.device("cpu")
# ------------------------------------------------------------------------
class SampleBuffer:
    def __init__(self, device):
        """
        A buffer for storing samples from Markov chain sampler, keeping the most
        probable sample for the next policy update.
        """
        self._device = device

    def update(self,states, logphis, counts, update_states, update_coeffs):
        self.states = states
        self.logphis = logphis
        self.counts = counts
        self.update_states = update_states
        self.update_coeffs = update_coeffs
        self._call_time = 0
        return

    def get(self, batch_size=100, batch_type='rand', sample_division=1):
        n_sample = len(self.states)
        devision_len = n_sample // sample_division 
        
        if n_sample <= batch_size:
            gpu_states = torch.from_numpy(self.states).to(self._device)
            gpu_counts = torch.from_numpy(self.counts).to(self._device)
            gpu_update_states = torch.from_numpy(self.update_states).to(self._device)
            gpu_update_coeffs = torch.from_numpy(self.update_coeffs).to(self._device)
            gpu_logphi0 = torch.from_numpy(self.logphis).to(self._device)
        elif batch_type == 'rand':
            batch_label = np.random.choice(n_sample, batch_size, replace=False)
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            counts = self.counts[batch_label]
            update_states = self.update_states[batch_label]
            update_coeffs = self.update_coeffs[batch_label]

            gpu_states = torch.from_numpy(states).to(self._device)
            gpu_counts = torch.from_numpy(counts).to(self._device)
            gpu_update_states = torch.from_numpy(update_states).to(self._device)
            gpu_update_coeffs = torch.from_numpy(update_coeffs).to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).to(self._device)
        elif batch_type == 'equal':
            if self._call_time < sample_division - 1:
                batch_label = range(self._call_time*devision_len, (self._call_time+1)*devision_len)
                self._call_time += 1
            else:
                batch_label = range(self._call_time*devision_len, n_sample)
                self._call_time = 0
            
            states = self.states[batch_label]
            logphis = self.logphis[batch_label]
            counts = self.counts[batch_label]
            update_states = self.update_states[batch_label]
            update_coeffs = self.update_coeffs[batch_label]

            gpu_states = torch.from_numpy(states).to(self._device)
            gpu_counts = torch.from_numpy(counts).to(self._device)
            gpu_update_states = torch.from_numpy(update_states).to(self._device)
            gpu_update_coeffs = torch.from_numpy(update_coeffs).to(self._device)
            gpu_logphi0 = torch.from_numpy(logphis).to(self._device)

        return dict(state=gpu_states, count=gpu_counts, update_states=gpu_update_states,
                    update_coeffs=gpu_update_coeffs, logphi0=gpu_logphi0)

# ------------------------------------------------------------------------
class train_Ops:
    def __init__(self, **kwargs):
        self._ham = kwargs.get('hamiltonian')
        self._get_init_state = kwargs.get('get_init_state')
        self._updator = kwargs.get('updator')
        # self._sampler = kwargs.get('sampler')
# ------------------------------------------------------------------------
# main training function
def train(epochs=100, Ops_args=dict(), Ham_args=dict(), n_sample=100, init_type='rand', n_optimize=10,
          learning_rate=1E-4, state_size=[10, 2], resample_condition=50, dimensions='1d', batch_size=1000,
          sample_division=5, target_wn=10, save_freq=10, net_args=dict(), threads=4, output_fn='test'):
    """
    main training process
    wavefunction: psi = phi*exp(1j*theta)
    output of the CNN network: logphi, theta

    Args:
        epochs (int): Number of epochs of interaction.

        n_sample (int): Number of sampling in each epoch.

        n_optimize (int): Number of update in each epoch.

        lr: learning rate for Adam.

        state_size: size of a single state, [n_sites, Dp].

        save_freq: frequency of saving.

        Dp: physical index.

        N or L, W: length of 1d lattice or length and with of 2d lattice
    """
    output_dir = os.path.join('./results', output_fn)
    save_dir = os.path.join(output_dir, 'save_model')
    logger = get_logger(os.path.join(output_dir, 'exp_log.txt'))

    if dimensions == '1d':
        TolSite = state_size[0]  # number of sites
        single_state_shape = [state_size[0]]
    else:
        TolSite = state_size[0]*state_size[1]
        single_state_shape = [state_size[0], state_size[1]]
    Dp = state_size[-1]  # number of physical spins

    train_ops = train_Ops(**Ops_args)
    _ham = train_ops._ham(**Ham_args)
    get_init_state = train_ops._get_init_state
    updator = train_ops._updator
    buffer = SampleBuffer(gpu)
    epsilon = np.min([0.01*learning_rate, 1E-4])

    logphi_model = mlp_cnn(state_size=state_size, output_size=2, **net_args)
    logger.info(logphi_model)
    logger.info(get_paras_number(logphi_model))

    MHsampler = MCsampler(state_size=state_size, model=logphi_model, init_type=init_type,
                          get_init_state=get_init_state, n_sample=n_sample, updator=updator, operator=_ham)

    # mean energy from importance sampling in GPU
    def _energy_ops(sample_division):
        data = buffer.get(batch_type='equal', sample_division=sample_division)
        states, counts, op_states, op_coeffs = data['state'], data['count'], data['update_states'], data['update_coeffs']

        with torch.no_grad():
            n_sample = op_states.shape[0]
            n_updates = op_states.shape[1]
            op_states = op_states.reshape([-1, Dp] + single_state_shape)

            psi_ops = logphi_model(op_states.float())
            logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
            theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

            psi = logphi_model(states.float())
            logphi = psi[:, 0].reshape(len(states), -1)
            theta = psi[:, 1].reshape(len(states), -1)

            delta_logphi_os = logphi_ops - logphi*torch.ones(logphi_ops.shape, device=gpu)
            delta_theta_os = theta_ops - theta*torch.ones(theta_ops.shape, device=gpu)
            Es = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)

            return (Es*counts).sum().to(cpu), ((Es**2)*counts).sum().to(cpu)

    # setting optimizer in GPU
    # optimizer = torch.optim.Adam(logphi_model.parameters(), lr=learning_rate)
    # imaginary time propagation: delta_t = learning_rate
    def update(IntCount, epsilon):
        data = buffer.get(batch_size=IntCount)
        logphi_model.zero_grad()

        state, count, op_states, op_coeffs = data['state'], data['count'], data['update_states'], data['update_coeffs']

        psi = logphi_model(state.float())
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)

        n_sample = op_states.shape[0]
        n_updates = op_states.shape[1]
        op_states = op_states.reshape([-1, Dp] + single_state_shape)
        psi_ops = logphi_model(op_states.float())
        logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
        theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

        delta_logphi_os = logphi_ops - logphi*torch.ones(logphi_ops.shape, device=gpu)
        delta_theta_os = theta_ops - theta*torch.ones(theta_ops.shape, device=gpu)
        ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1)
        ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1)
        
        with torch.no_grad():
            ops = ops_real.float() + 1j*ops_imag.float() # batch_size
            mean_e = (ops*count).sum(0)/count.sum()
        
        # update parameters with gradient descent
        for param in logphi_model.parameters():
            param_len = len(param.data.reshape(-1))
            Oks = torch.zeros([n_sample, param_len], device=gpu, dtype=torch.cfloat)
            Oks_conj = torch.zeros([n_sample, param_len], device=gpu, dtype=torch.cfloat)
            OO_matrix = torch.zeros([n_sample, param_len, param_len], device=gpu, dtype=torch.cfloat)

            # calculate stochastic reconfiguration matrix Skk
            grad_outputs = torch.zeros_like(logphi)
            for i in range(n_sample): # for-loop over a batch
                grad_outputs[i] = 1
                grads_real = gradient(logphi, param, grad_outputs=grad_outputs).reshape(1, param_len)
                grads_imag = gradient(theta, param, grad_outputs=grad_outputs).reshape(1, param_len)
                with torch.no_grad():
                    Oks[i] = (grads_real + 1j*grads_imag)*count[i]
                    Oks_conj[i] = (grads_real - 1j*grads_imag)*count[i]
                    OO_matrix[i] = (Oks_conj[i][..., None]*Oks[i])/count[i]
                    # test_m = OO_matrix[i]
                    # test_v = (test_m - test_m.conj().t()).sum()
                    # print((test_m - test_m.conj().t()).sum())
                    #if test_v.abs() > 0:
                    #    print(Oks[i])
                    #    print(OO_matrix[i])
            # print(OO_matrix.shape)
            Skk_matrix = OO_matrix.sum(0)/count.sum() - (Oks_conj.sum(0)[..., None]/count.sum())*(Oks.sum(0)/count.sum()) 
            # print([OO_matrix.sum(0).shape, ((Oks_conj.sum(0)/count.sum())*(Oks.sum(0).reshape(-1,1)/count.sum())).shape])
            # calculate Fk
            Fk = (ops[...,None]*Oks_conj).sum(0)/count.sum() - mean_e*(Oks_conj.sum(0)/count.sum())
            # calculateprint(Fk.real.float().to(cpu).numpy())
            Skk_inv = torch.linalg.pinv(Skk_matrix + epsilon*torch.eye(Skk_matrix.shape[0], device=gpu))
            # print(torch.diag(Skk_matrix))
            update_k = torch.matmul(Skk_inv, Fk).real
            update_k = torch.clamp(update_k, min=-1000, max=1000)
            # print(update_k)
            # print([ops.shape, (mean_e*(Oks_conj.sum(0).reshape(-1,1)/n_sample)).shape, param_len, param.data.shape, update_k.shape])
            # update the parameters
            # x = param.data.clone()
            # param.data -= learning_rate*Fk.real.float().reshape(param.data.shape)
            param.data -= learning_rate*update_k.reshape(param.data.shape)
        
        return mean_e.real.to(cpu).numpy()

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('Start training:')
    warmup_n_sample = n_sample // 5

    for epoch in range(epochs):
        sample_tic = time.time()
        MHsampler._n_sample = warmup_n_sample
        states, logphis, update_states, update_coeffs = MHsampler.parallel_mh_sampler(threads)
        n_sample = MHsampler._n_sample

        # using unique states to reduce memory usage.
        states, logphis, counts, update_states, update_coeffs = _get_unique_states(states, logphis,
                                                                            update_states, update_coeffs)

        buffer.update(states, logphis, counts, update_states, update_coeffs)

        IntCount = len(states)

        sample_toc = time.time()

        logphi_model = logphi_model.to(gpu)
        # ------------------------------------------GPU------------------------------------------
        op_tic = time.time()
        # epsilon_decay = epsilon*(0.9**(epoch//50))
        mean_e = update(IntCount, epsilon)
        # logger.info(mean_e.to(cpu).detach().numpy()/TolSite)
        op_toc = time.time()
        
        
        sd = 1 if IntCount < batch_size else sample_division
        avgE = torch.zeros(sd)
        avgE2 = torch.zeros(sd)
        for i in range(sd):    
            avgE[i], avgE2[i] = _energy_ops(sd)
        # ---------------------------------------------------------------------------------------
        logphi_model = logphi_model.to(cpu)

        # average over all samples
        AvgE = avgE.sum().numpy()/n_sample
        AvgE2 = avgE2.sum().numpy()/n_sample
        StdE = np.sqrt(AvgE2 - AvgE**2)/TolSite
        Dloss = AvgE - mean_e

        # print training informaition
        logger.info('Epoch: {}, AvgE: {:.5f}, StdE: {:.5f}, Dloss: {:.3f}, IntCount: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
                    format(epoch, AvgE/TolSite, StdE, Dloss, IntCount, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))

        # save the trained NN parameters
        if epoch % save_freq == 0 or epoch == epochs - 1:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            torch.save(logphi_model.state_dict(), os.path.join(
                save_dir, 'model_'+str(epoch)+'.pkl'))

        if warmup_n_sample != n_sample:
            # first 5 epochs are used to warm up due to the limitations of memory
            warmup_n_sample += n_sample // 5

    logger.info('Finish training.')

    return logphi_model, AvgE
