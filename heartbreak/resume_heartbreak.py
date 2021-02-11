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
def trpo_gym_tf_music(ctxt=None, saved_dir ='./', seed=1):
    """Train TRPO with Music-v0 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
            created by @wrap_experiment
        seed (int): Used to seed the random number generator to produce
            determinism.

    """


    set_seed(seed)
    ctxt.snapshot_mode = 'none'

    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(from_dir=saved_dir)
        trainer.resume(n_epochs=120, batch_size=1, store_episodes = True)

saved_dir = './data/local/experiment/trpo_gym_tf_music_54/'
trpo_gym_tf_music(saved_dir = saved_dir)
