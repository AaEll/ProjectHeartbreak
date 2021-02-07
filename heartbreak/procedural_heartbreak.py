#!/usr/bin/env python3
"""An example to train a task with TRPO algorithm."""
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.tf.baselines import GaussianMLPBaseline
from garage.sampler import LocalSampler
from garage.replay_buffer import PathBuffer
from garage.tf.algos import NPO
from garage.experiment.experiment import ExperimentContext
from garage.tf.policies import GaussianLSTMPolicy, CategoricalMLPPolicy
from garage.trainer import TFTrainer
from gym_music.envs import MusicEnv
from gym_music.utils.monitors import HeartMonitor
import gym
import garage


ctxt = ExperimentContext( snapshot_dir = "./", snapshot_mode = 'none', snapshot_gap = 3)
seed = 2021

set_seed(seed)

trainer = TFTrainer(snapshot_config=ctxt)
env = GymEnv(MusicEnv(monitor = HeartMonitor('DC:39:39:66:26:1F')),max_episode_length = 25) 
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
        
algo = NPO(env_spec = env.spec,
             policy = policy,
             baseline = baseline, 
             sampler = sampler,
            )
        
trainer.setup(algo, env)
print("trainer.train(n_epochs=120, batch_size=1)")


