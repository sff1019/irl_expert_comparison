import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import MultiprocessingSampler
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
import torch

from torchrl.algos import GAIL, TRPO
from torchrl.experiments import IRLTrainer
from torchrl.utils.misc import load_latest_experts

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--log_dir', type=str, default='data/gail_pendulum_v0')
parser.add_argument('--experts_dir', type=str, default='data/trpo_pendulum')
args = parser.parse_args()


@wrap_experiment(log_dir=args.log_dir,
                 snapshot_mode='all',
                 archive_launch_repo=False)
def main(ctxt=None, seed=0):
    env = GymEnv('Pendulum-v0')

    print('here env')
    experts = load_latest_experts(args.experts_dir, n=5)

    print('here experts')
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                discount=0.99,
                center_adv=False)

    trainer = IRLTrainer(ctxt)
    trainer.setup(algo,
                  env,
                  irl_model,
                  baseline,
                  n_itr=args.n_iter,
                  sampler_cls=MultiprocessingSampler,
                  zero_environment_reward=True)
    trainer.train(n_epochs=1, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
