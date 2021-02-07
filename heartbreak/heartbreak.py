#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.tf.baselines import GaussianMLPBaseline
from garage.sampler import LocalSampler
from garage.replay_buffer import PathBuffer
from garage.tf.algos import NPO
from garage.tf.policies import GaussianLSTMPolicy, CategoricalMLPPolicy
from garage.trainer import TFTrainer
from gym_music.envs import MusicEnv
from gym_music.utils.monitors import HeartMonitor
import gym


@wrap_experiment
def trpo_gym_tf_music(ctxt=None, seed=1):
    """Train TRPO with Music-v0 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
            created by @wrap_experiment
        seed (int): Used to seed the random number generator to produce
            determinism.

    """

    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:

        #env = GymEnv(gym.make('Pendulum-v0')) #
        env = GymEnv(MusicEnv(monitor = HeartMonitor('DC:39:39:66:26:1F')),max_episode_length = 35) 
        policy = GaussianLSTMPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_dim= 32)
        
        baseline = GaussianMLPBaseline(
            env_spec = env.spec,
            hidden_sizes=(32, 32),
        )

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=False,
                               n_workers = 1,
                              )
        #replay_buffer = PathBuffer( env_spec = env.spec, capacity_in_transitions = 2000) 
        
        algo = NPO(env_spec = env.spec,
                    policy = policy,
                    baseline = baseline, 
                    sampler = sampler,
                  )
        
        trainer.setup(algo, env)

        print("!!!")
        trainer.train(n_epochs=120, batch_size=1)
        print("!")

trpo_gym_tf_music()
