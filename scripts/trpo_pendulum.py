import argparse

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import MultiprocessingSampler
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
import torch

from torchrl.algos.trpo import TRPO
from torchrl.experiments.trainer import Trainer


@wrap_experiment(log_dir='data/trpo_pendulum_v0', snapshot_mode='all')
def main(ctxt=None, seed=0):
    env = GymEnv('Pendulum-v0')

    policy = GaussianMLPPolicy(env.spec, hidden_sizes=[32, 32])

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                discount=0.99,
                center_adv=False)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env, sampler_cls=MultiprocessingSampler)
    trainer.train(n_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main()
