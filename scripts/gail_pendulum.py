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


@wrap_experiment(log_dir='data/gail_pendulum_v0', snapshot_mode='all')
def main(ctxt=None, seed=0):
    env = GymEnv('InvertedPendulum-v2')

    experts = load_latest_experts('data/trpo_pendulum', n=5)

    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
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

    trainer = IRLTrainer(ctxt)
    trainer.setup(algo,
                  env,
                  irl_model,
                  baseline,
                  sampler_cls=MultiprocessingSampler,
                  zero_environment_reward=True)
    trainer.train(n_epochs=1, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main()
