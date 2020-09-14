import copy
import os
import time

import cloudpickle
from dowel import logger, tabular
import psutil

# This is avoiding a circular import
from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.snapshotter import Snapshotter
from garage.sampler.default_worker import DefaultWorker
from garage.sampler.worker_factory import WorkerFactory

# pylint: disable=no-name-in-module


class ExperimentStats:
    # pylint: disable=too-few-public-methods
    """Statistics of a experiment.

    Args:
        total_epoch (int): Total epoches.
        total_itr (int): Total Iterations.
        total_env_steps (int): Total environment steps collected.
        last_episode (list[dict]): Last sampled episodes.

    """
    def __init__(self, total_epoch, total_itr, total_env_steps, last_episode):
        self.total_epoch = total_epoch
        self.total_itr = total_itr
        self.total_env_steps = total_env_steps
        self.last_episode = last_episode


class SetupArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to setup a trainer.

    Args:
        sampler_cls (Sampler): A sampler class.
        sampler_args (dict): Arguments to be passed to sampler constructor.
        seed (int): Random seed.

    """
    def __init__(self, sampler_cls, sampler_args, seed):
        self.sampler_cls = sampler_cls
        self.sampler_args = sampler_args
        self.seed = seed


class TrainArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to call train() or resume().

    Args:
        n_epochs (int): Number of epochs.
        batch_size (int): Number of environment steps in one batch.
        plot (bool): Visualize an episode of the policy after after each epoch.
        store_episodes (bool): Save episodes in snapshot.
        pause_for_plot (bool): Pause for plot.
        start_epoch (int): The starting epoch. Used for resume().

    """
    def __init__(self, n_epochs, batch_size, plot, store_episodes,
                 pause_for_plot, start_epoch):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.plot = plot
        self.store_episodes = store_episodes
        self.pause_for_plot = pause_for_plot
        self.start_epoch = start_epoch


class IRLTrainer(Trainer):
    def __init__(self, snapshot_config):
        super(IRLTrainer, self).__init__()
        self._snapshotter = Snapshotter(snapshot_config.snapshot_dir,
                                        snapshot_config.snapshot_mode,
                                        snapshot_config.snapshot_gap)

        self._has_setup = False
        self._plot = False

        self._setup_args = None
        self._train_args = None
        self._stats = ExperimentStats(total_itr=0,
                                      total_env_steps=0,
                                      total_epoch=0,
                                      last_episode=None)

        self._algo = None
        self._env = None
        self._sampler = None
        self._plotter = None

        self._start_time = None
        self._itr_start_time = None
        self.step_itr = None
        self.step_episode = None

        # only used for off-policy algorithms
        self.enable_logging = True

        self._n_workers = None
        self._worker_class = None
        self._worker_args = None

    @override
    def setup(self,
              algo,
              env,
              irl,
              sampler_cls=None,
              sampler_args=None,
              n_workers=psutil.cpu_count(logical=False),
              worker_class=None,
              worker_args=None):
        self._algo = algo
        self._env = env
        self._irl = irl
        self._n_workers = n_workers
        self._worker_class = worker_class
        if sampler_args is None:
            sampler_args = {}
        if sampler_cls is None:
            sampler_cls = getattr(algo, 'sampler_cls', None)
        if worker_class is None:
            worker_class = getattr(algo, 'worker_cls', DefaultWorker)
        if worker_args is None:
            worker_args = {}

        self._worker_args = worker_args
        if sampler_cls is None:
            self._sampler = None
        else:
            self._sampler = self.make_sampler(sampler_cls,
                                              sampler_args=sampler_args,
                                              n_workers=n_workers,
                                              worker_class=worker_class,
                                              worker_args=worker_args)

        self._has_setup = True

        self._setup_args = SetupArgs(sampler_cls=sampler_cls,
                                     sampler_args=sampler_args,
                                     seed=get_seed())

    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        if self._sampler is None:
            raise ValueError('trainer was not initialized with `sampler_cls`. '
                             'Either provide `sampler_cls` to trainer.setup, '
                             ' or set `algo.sampler_cls`.')
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError(
                'trainer was not initialized with `batch_size`. '
                'Either provide `batch_size` to trainer.train, '
                ' or pass `batch_size` to trainer.obtain_samples.')
        episodes = None
        if agent_update is None:
            agent_update = self._algo.policy.get_param_values()
        episodes = self._sampler.obtain_samples(
            itr, (batch_size or self._train_args.batch_size),
            agent_update=agent_update,
            env_update=env_update)
        self._stats.total_env_steps += sum(episodes.lengths)
        return episodes

    def obtain_samples(self,
                       itr,
                       batch_size=None,
                       agent_update=None,
                       env_update=None):
        eps = self.obtain_episodes(itr, batch_size, agent_update, env_update)
        return eps.to_list()

    def save(self, epoch):
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['setup_args'] = self._setup_args
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        params['env'] = self._env
        params['algo'] = self._algo
        params['irl'] = self._irl
        params['n_workers'] = self._n_workers
        params['worker_class'] = self._worker_class
        params['worker_args'] = self._worker_args

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')

    def restore(self, from_dir, from_epoch='last'):
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._setup_args = saved['setup_args']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._setup_args.seed)

        self.setup(env=saved['env'],
                   algo=saved['algo'],
                   sampler_cls=self._setup_args.sampler_cls,
                   sampler_args=self._setup_args.sampler_args,
                   n_workers=saved['n_workers'],
                   worker_class=saved['worker_class'],
                   worker_args=saved['worker_args'])

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        batch_size = self._train_args.batch_size
        store_episodes = self._train_args.store_episodes
        pause_for_plot = self._train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_episodes', store_episodes))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))
        logger.log(fmt.format('-- Stats --', '-- Value --'))
        logger.log(fmt.format('last_itr', last_itr))
        logger.log(fmt.format('total_env_steps', total_env_steps))

        self._train_args.start_epoch = last_epoch + 1
        return copy.copy(self._train_args)

    def _train_irl(self, samples, itr=0):
        if self.no_reward:
            total_rew = 0.
            for path in paths:
                tot_rew += np.sum(path['rewards'])
                path['rewards'] *= 0
            tabular.record('OriginalTaskAverageReturn',
                           total_rew / float(len(paths)))

        if self.irl_model_wt <= 0:
            return paths

        if self.train_irl:
            max_iters = self.discrim_train_iters
            mean_loss = self.irl_model.train(
                paths,
                policy=self.policy,
                itr=itr,
                batch_size=self.discrim_batch_size,
                max_itrs=max_iters,
                logger=logger)

            tabular.record('IRLLoss', mean_loss)
            self.irl_params = self.irl_model.get_params()

        estimated_rewards = self.irl_model.eval(paths,
                                                gamma=self.discount,
                                                itr=itr)

        tabular.record('IRLRewardMean',
                       np.mean(np.concatenate(estimated_rewards)))
        tabular.record('IRLRewardMean',
                       np.max(np.concatenate(estimated_rewards)))
        tabular.record('IRLRewardMean',
                       np.min(np.concatenate(estimated_rewards)))

        # Replace the original reward signal with learned reward signal
        # This will be used by agents to learn policy
        if self.irl_model.score_trajectories:
            for i, path in enumerate(paths):
                path['rewards'][-1] += self.irl_model_wt * estimated_rewards[i]
        else:
            for i, path in enumerate(paths):
                path['rewards'] += self.irl_model_wt * estimated_rewards[i]
        return paths

    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore for algo
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot

        returns = []
        for itr in range(self.start_itr, self.n_itr):
            with logger.prefix(f'itr #{itr} | '):
                logger.log('Obtaining paths...')
                paths = self.obtain_paths(itr)
                logger.log('Processing paths...')

                # compute irl and update reward function
                paths = self.compute_irl(paths, itr=itr)
                samples_data = self.process_paths(itr, paths)

                logger.log('Logging diagnostics...')
                self.log_diagnostics(paths)
                logger.log('Optimizing policy...')

                # train policy
                self._algo.train(self)
                logger.log('Saving snapshot...')
                self.save(itr)
                logger.log('Saved')
                tabular.record('Time', time.time() - start_time)
                tabular.record('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)

        self._shutdown_worker()

        return

    def step_epochs(self):
        self._start_worker()
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_episode = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch
                save_episode = (self.step_episode
                                if self._train_args.store_episodes else None)

                self._stats.last_episode = save_episode
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)

                if self.enable_logging:
                    self.log_diagnostics(self._train_args.pause_for_plot)
                    logger.dump_all(self.step_itr)
                    tabular.clear()

    def get_env_copy(self):
        """Get a copy of the environment.

        Returns:
            Environment: An environment instance.

        """
        if self._env:
            return cloudpickle.loads(cloudpickle.dumps(self._env))
        else:
            return None

    def process_samples(self, itr, paths):
        baselines, returns = [], []

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list(
                [path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list(
                [path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list(
                [path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list(
                [path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list(
                [path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list(
                [path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list(
                [path["agent_infos"] for path in paths])

            return dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )
        else:
            print("MUST IMPLEMENT")
            exit()

    @property
    def total_env_steps(self):
        """Total environment steps collected.

        Returns:
            int: Total environment steps collected.

        """
        return self._stats.total_env_steps


class NotSetupError(Exception):
    """Raise when an experiment is about to run without setup."""
