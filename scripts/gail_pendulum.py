import argparse

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import torch

from torchrl.utils.misc import load_latest_experts
from torchrl.algos import GAIL, TRPO


# @wrap_experiment(log_dir='data/trpo_pendulum', snapshot_mode='all')
def main(ctxt=None, seed=0):
    env = GymEnv('InvertedPendulum-v2')

    experts = load_latest_experts('data/trpo_pendulum', n=5)

    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    # policy = GaussianMLPPolicy(env.spec, hidden_sizes=[32, 32])
    #
    # value_function = GaussianMLPValueFunction(env_spec=env.spec,
    #                                           hidden_sizes=(32, 32),
    #                                           hidden_nonlinearity=torch.tanh,
    #                                           output_nonlinearity=None)
    #
    # algo = TRPO(env_spec=env.spec,
    #             policy=policy,
    #             value_function=value_function,
    #             discount=0.99,
    #             center_adv=False)
    #
    # trainer = Trainer(ctxt)
    # trainer.setup(algo, env)
    # trainer.train(n_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main()
