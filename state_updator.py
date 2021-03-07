# encoding: utf-8

import numpy as np

class updator():
    def __init__(self, state_size):
        self._N = state_size[0]
        self._Dp = state_size[1]

    def generate_mask(self, n_sample):
        rang = range(n_sample)
        swaps = np.random.randint(0, self._N, (2, n_sample))
        masks = np.arange(self._N)[None, :].repeat(n_sample, axis=0)
        masks[rang, swaps[0]], masks[rang, swaps[1]] = (
            masks[rang, swaps[1]], masks[rang, swaps[0]])
        # print(np.random.randint(0,10))
        return masks

    def _get_update(self, state, mask):
        self._state = state.T
        self._state = self._state[mask]
        # self._state = np.concatenate((self._state, self._state[0,:].reshape(1,self._Dp)),0)
        return self._state.T

if __name__ == "__main__":
    from nqs_vmc_torch1d import _get_init_nqs
    state0, ms = _get_init_nqs(10,2,kind='rand')
    print(ms)
    state0 = state0.reshape(2,10)
    print(state0)
    Update = updator([10,2])
    masks = Update.generate_mask(100)
    print(masks[10])
    statef = Update._get_update(state0,masks[10])
    print(statef)
    # print(statef)
    # print(statef - state0)
