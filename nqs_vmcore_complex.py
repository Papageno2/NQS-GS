# encoding: utf-8
# version 1.0 updates: parallel sampling, logger, saves
# version0 2.0 updates: compatible with 2d system, replay buffer

import numpy as np
import torch
import torch.nn as nn
from mcmc_sampler_complexv2 import MCsampler
from core import mlp_cnn, get_paras_number
from utils import get_logger
import time
import os

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
# ------------------------------------------------------------------------
def _get_unique_states(states, logphis, ustates, ucoeffs):
    """
    Returns the unique states, their coefficients and the counts.
    """
    states, indices, counts = np.unique(states, return_index=True, return_counts=True, axis=0)
    logphis = logphis[indices]
    ustates = ustates[indices]
    ucoeffs = ucoeffs[indices]
    return states, logphis, counts, ustates, ucoeffs

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
        
        if n_sample < batch_size:
            batch_size = n_sample
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
          learning_rate=1E-4, state_size=[10, 2], resample_condition=50, dimensions='1d', batch_size=500,
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

    # define the loss function according to the energy functional in GPU
    def compute_loss_energy(data):
        # no_grad for weights, op_coeffs and mean_energy
        # state: states from the sampling
        # count: number of the sampling
        # logphi0, theta0: output of the NN before optimization
        state, count, op_states = data['state'], data['count'], data['update_states']
        op_coeffs, logphi0 = data['update_coeffs'], data['logphi0']

        psi = logphi_model(state.float())
        logphi = psi[:, 0].reshape(len(state), -1)
        theta = psi[:, 1].reshape(len(state), -1)

        # calculate the weights of the energy from important sampling
        delta_logphi = logphi - logphi0[..., None]

        # delta_logphi = delta_logphi - delta_logphi.mean()*torch.ones(delta_logphi.shape)
        delta_logphi = delta_logphi - delta_logphi.mean()
        weights = count[..., None]*torch.exp(delta_logphi * 2)
        weights_norm = weights.sum()
        weights = (weights/weights_norm).detach()

        # calculate the coeffs of the energy
        n_sample = op_states.shape[0]
        n_updates = op_states.shape[1]
        op_states = op_states.reshape([-1, Dp]+single_state_shape)
        psi_ops = logphi_model(op_states.float())
        logphi_ops = psi_ops[:, 0].reshape(n_sample, n_updates)
        theta_ops = psi_ops[:, 1].reshape(n_sample, n_updates)

        delta_logphi_os = logphi_ops - logphi*torch.ones(logphi_ops.shape, device=gpu)
        delta_theta_os = theta_ops - theta*torch.ones(theta_ops.shape, device=gpu)
        ops_real = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.cos(delta_theta_os), 1).detach()
        ops_imag = torch.sum(op_coeffs*torch.exp(delta_logphi_os)*torch.sin(delta_theta_os), 1).detach()

        # calculate the mean energy
        mean_energy_real = (weights*ops_real[..., None]).sum().detach()
        mean_energy_imag = (weights*ops_imag[..., None]).sum().detach()

        loss_e = ((weights*ops_real[..., None]*logphi).sum() - mean_energy_real*(weights*logphi).sum()
                  + (weights*ops_imag[..., None]*theta).sum() - mean_energy_imag*(weights*theta).sum())

        return loss_e, mean_energy_real, weights_norm/count.sum()

    # setting optimizer in GPU
    optimizer = torch.optim.Adam(logphi_model.parameters(), lr=learning_rate)

    # off-policy optimization from li yang
    def update():
        data = buffer.get(batch_size=batch_size)
        loss_e_old, _, _ = compute_loss_energy(data)
        # off-policy update
        mean_e_tol = 0
        wn_tol = 0
        es = 0
        for i in range(n_optimize):
            optimizer.zero_grad()
            loss_e, mean_e, wn = compute_loss_energy(data)
            mean_e_tol += mean_e
            wn_tol += wn

            if wn > target_wn:
                logger.warning(
                    'early stop at step={} as reaching maximal WsN'.format(i))
                es = 1
                break
            loss_e.backward()
            optimizer.step()

        return loss_e_old-loss_e, wn_tol/(i+1), es

    # ----------------------------------------------------------------
    tic = time.time()
    logger.info('Start training:')
    warmup_n_sample = n_sample // 5
    es_cnt = 0

    for epoch in range(epochs):
        sample_tic = time.time()
        MHsampler._n_sample = warmup_n_sample
        states, logphis, _, update_states, update_coeffs = MHsampler.parallel_mh_sampler(threads)
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
        Dloss, WsN, es = update()
        op_toc = time.time()
        es_cnt += es

        sd = 1 if IntCount < batch_size else sample_division
        avgE = torch.zeros(sd)
        avgE2 = torch.zeros(sd)
        for i in range(sd):    
            avgE[i], avgE2[i] = _energy_ops(sd)
        # ---------------------------------------------------------------------------------------
        Dloss, WsN = Dloss.to(cpu), WsN.to(cpu)
        logphi_model = logphi_model.to(cpu)

        # average over all samples
        AvgE = avgE.sum().numpy()/n_sample
        AvgE2 = avgE2.sum().numpy()/n_sample
        StdE = np.sqrt(AvgE2 - AvgE**2)/TolSite

        # print training informaition
        logger.info('Epoch: {}, AvgE: {:.5f}, StdE: {:.5f}, Dloss: {:.3f}, WsN: {:.3f}, IntCount: {}, SampleTime: {:.3f}, OptimTime: {:.3f}, TolTime: {:.3f}'.
                    format(epoch, AvgE/TolSite, StdE, Dloss, WsN, IntCount, sample_toc-sample_tic, op_toc-op_tic, time.time()-tic))

        # resample the initial state
        if es_cnt > resample_condition:
            logger.info('resample the initial states')
            MHsampler.resample_init_states(init_type='ferro', threads=threads)
            es_cnt = 0

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
