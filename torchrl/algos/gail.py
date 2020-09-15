"""
Implementation of Generative Adversarial Imitation Learning (GAIL).
url: https://arxiv.org/abs/1606.03476
"""

from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
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

    def predict(self, obs, acts):
        obs = Variable(torch.from_numpy(obs), volatile=True).float()
        acts = Variable(torch.from_numpy(acts)).float()

        inputs = torch.cat((obs, acts), axis=1)
        outputs = self.forward(inputs)
        probs = F.sigmoid(outputs)

        return probs.data.numpy()

    def get_params(self):
        return self.parameters()

    def get_named_params(self):
        return self.named_parameters()


class GAIL(object):
    def __init__(self,
                 env_spec,
                 hidden_size=[32, 32],
                 expert_trajs=None,
                 batch_size=32,
                 lr=1.e-3,
                 favor_zero_expert_reward=True):
        super(GAIL, self).__init__()
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.expert_trajs = self.extract_paths(expert_trajs)
        self.favor_zero_expert_reward = favor_zero_expert_reward

        self.disc = Discriminator(self.obs_dim,
                                  self.action_dim,
                                  hidden_size=hidden_size)

        self.optim = optim.Adam(self.disc.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size

    def _compute_path_probs(paths, insert=True, insert_key='a_logprobs'):
        """
        Returns a N * T matrix of action probabilities, where N is number
        of trajectories, and T is the length of each trajectories.
        """
        policy_dist_type = 'gaussian'

        n_path = len(paths)
        actions = [path['actions'] for path in paths]
        params = [(path['agent_infos']['mean'], path['agent_infos']['log_std'])
                  for path in paths]
        path_probs = [
            gauss_log_pdf(params[i], actions[i]) for i in range(num_path)
        ]

        if insert:
            for i, path in enumerate(paths):
                path[insert_key] = path_probs[i]

        return np.array(path_probs)

    @property
    def score_trajectories(self):
        return False

    @staticmethod
    def extract_paths(paths, keys=['observations', 'actions'], stack=False):
        if stack:
            for key in keys:
                for p in paths:
                    print(p[key].shape)
            return [
                np.stack([t[key] for t in paths]).astype(np.float32)
                for key in keys
            ]
        else:
            return [
                np.concatenate([t[key] for t in paths]).astype(np.float32)
                for key in keys
            ]

    def sample_batch(self, *args):
        batch_idxs = np.random.randint(
            0, args[0].shape[0], self.batch_size)  # trajectories are negatives
        return [data[batch_idxs] for data in args]

    @staticmethod
    def unpack(data, paths):
        lengths = [path['observations'].shape[0] for path in paths]

        unpacked = []
        idx = 0
        for l in lengths:
            unpacked.append(data[idx:idx + l])
            idx += l

        return unpacked

    def train(self, trajs, max_itrs=100):
        agt_states, agt_acts = self.extract_paths(trajs)
        exp_states, exp_acts = self.expert_trajs

        total_loss = 0.
        # train discriminator
        for itr in range(max_itrs):
            agt_states_batch, agt_act_batch = self.sample_batch(
                agt_states, agt_acts)
            exp_states_batch, exp_act_batch = self.sample_batch(
                exp_states, exp_acts)

            # create labels
            labels = np.zeros((self.batch_size * 2, 1))
            labels[self.batch_size:] = 1.0

            states_batch = np.concatenate([agt_states_batch, exp_states_batch])
            act_batch = np.concatenate([agt_act_batch, exp_act_batch])

            # convert to torch.tensor
            states_batch = Variable(torch.from_numpy(states_batch)).type(
                torch.FloatTensor)

            act_batch = Variable(torch.from_numpy(act_batch)).type(
                torch.FloatTensor)
            labels = Variable(torch.from_numpy(labels)).type(torch.FloatTensor)

            inputs = torch.cat((states_batch, act_batch), axis=1)

            self.optim.zero_grad()
            out = self.disc(inputs)
            loss = self.loss_fn(out, labels)
            loss.backward()
            self.optim.step()

            total_loss += loss

        return total_loss / max_itrs

    def eval(self, paths, eps=1.e-8, **kwargs):
        obs, acts = self.extract_paths(paths)
        scores = self.disc.predict(obs, acts)

        if self.favor_zero_expert_reward:
            scores = np.log(scores[:, 0] + eps)
        else:
            scores = -np.log(1 - scores[:, 0] + eps)

        return self.unpack(scores, paths)

    def get_params(self):
        return self.disc.get_params()
