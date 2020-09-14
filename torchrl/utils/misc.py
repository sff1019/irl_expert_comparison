import contextlib
import joblib
import json
import os
import random
import re

import numpy as np


def extract(x, *keys):
    if isinstance(x, (dict, LazyDict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def get_expert_fnames(log_dir, n=5):
    print('Looking for paths')
    itr_reg = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    itr_files = []
    for i, filename in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('itr_count')
            itr_files.append((itr_count, filename))

    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n]
    for itr_file_and_count in itr_files:
        fname = os.path.join(log_dir, itr_file_and_count[1])
        print('Loading %s' % fname)
        yield fname


def load_experts(fname,
                 max_files=float('inf'),
                 min_return=None,
                 include_next_state=False):
    if hasattr(fname, '__iter__'):
        paths = []
        for fname_ in fname:
            snapshot_dict = joblib.load(fname_)
            paths.extend(snapshot_dict['paths'])
    else:
        snapshot_dict = joblib.load(fname)
        paths = snapshot_dict['paths']

    trajs = []
    for path in paths:
        obses = path['observations']
        actions = path['actions']
        returns = path['returns']
        total_return = np.sum(returns)
        if (min_return is None) or (total_return >= min_return):
            if include_next_state:
                next_obs = path['next_obs']
                terminals = path['terminals']
                traj = {
                    'observations': obses,
                    'actions': actions,
                    'next_obs': next_obs,
                    'terminals': terminals
                }
            else:
                traj = {'observations': obses, 'actions': actions}
            trajs.append(traj)
    random.shuffle(trajs)
    print('Loaded %d trajectories' % len(trajs))
    return trajs


def load_latest_experts(logdir,
                        n=5,
                        min_return=None,
                        include_next_state=False):
    """
    Load the trajectories from the last n epoches of training of expert.
    Parameters
    ----------
    n (int): Indicate the last n epoches we want to load trajectories from.
    min_return (float): A minimum reward threshold for a trajectory to qualify
        as demonstration.
    include_next_state (boolean): Do we use (s, a, r) tuple or
        (s, a, r, s', t) tuple
    """
    return load_experts(get_expert_fnames(logdir, n=n),
                        min_return=min_return,
                        include_next_state=include_next_state)
