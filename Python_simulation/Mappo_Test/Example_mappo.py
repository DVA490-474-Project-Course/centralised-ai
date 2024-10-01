
import torch
import torch.nn as nn
from EnvFootball import FootballEnv
# import the skrl components to build the RL system
from skrl.envs.loaders.torch import load_bidexhands_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed



env = FootballEnv()
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device='cpu', replacement=False)


# instantiate the agent's models (function approximators).
# MAPPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#models
models = {}


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#configuration-and-hyperparameters
cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 6  # 24 * 4096 / 16384
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 180
cfg["experiment"]["checkpoint_interval"] = 1800
cfg["experiment"]["directory"] = "runs/torch/ShadowHandOver"

agent = MAPPO(possible_agents=env.possible_agents,
              models=models,
              memories=memory,
              cfg=cfg,
              observation_spaces=env.observation_spaces,
              action_spaces=env.action_space,
              device='cpu',
              shared_observation_spaces=env.shared_observation_spaces)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 36000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()