"""
Implementation of Generative Adversarial Imitation Learning (GAIL).
url: https://arxiv.org/abs/1606.03476
"""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_size=[32, 32],
                 hidden_act_fn=nn.ReLU):
        super(Discriminator, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        dims = [obs_dim + action_dim] + list(hidden_size)
        for idx, dim in enumerate(dims):
            if idx != len(dims) - 1:
                layers.append(nn.Linear(dim, dims[idx + 1]))
                layers.append(hidden_act_fn())
            else:
                layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class ImitationLearning:
    pass


class SingleTimestepIRL(ImitationLearning):
    pass


class GAIL():
    def __init__(self,
                 env_spec,
                 hidden_size=[32, 32],
                 expert_trajs=None,
                 batch_size=32,
                 lr=1.e-3):
        super(GAIL, self).__init__()
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.expert_trajs = expert_trajs

        self.disc = Discriminator(self.obs_dim_dim,
                                  self.action_dim,
                                  hidden_size=hidden_size)

        self.optim = optim.Adam(disc.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size

    def train(self, trajs):
        agt_states, agt_acts = self.extract_paths(trajs)
        exp_states, exp_acts = self.expert_trajs

        total_loss = 0.
        for itr in range(max_itrs):
            agt_states_batch, agt_act_batch = self.sample_batch(
                agt_states, agt_acts)
            exp_states_batch, exp_act_batch = self.sample_batch(
                exp_states, exp_acts)

            # create labels
            labels = np.zeros((self.batch_size * 2, 1))
            labels[batch_size:] = 1.0

            states_batch = np.concatenate([agt_states_batch, exp_states_batch])
            act_batch = np.concatenate([agt_act_batch, exp_act_batch])

            # convert to torch.tensor
            states_batch = torch.from_numpy(states_batch).type(
                torch.FloatTensor)
            act_batch = torch.from_numpy(act_batch).type(torch.FloatTensor)
            labels = torch.from_numpy(labels).type(torch.FloatTensor)

            inputs = torch.cat((states_batch, act_batch), axis=1)

            self.optim.zero_grad()
            out = self.disc(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()

            total_loss += loss

        return total_loss /= max_itrs

    def eval(self):
        pass
