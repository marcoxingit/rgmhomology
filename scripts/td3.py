#
#  This script trains an "expert" on DeepMind Control Suite (DMC) environments using TD3.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from dm_control import suite
from dm_wrapper import DMEnv

from skrl.models.torch import DeterministicMixin, Model
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer

from argparse import ArgumentParser


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.silu(
            F.layer_norm(
                self.linear_layer_1(inputs["states"]),
                [
                    400,
                ],
            )
        )
        x = F.silu(
            F.layer_norm(
                self.linear_layer_2(x),
                [
                    300,
                ],
            )
        )
        return torch.tanh(self.action_layer(x)), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.silu(
            F.layer_norm(
                self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)),
                [
                    400,
                ],
            )
        )
        x = F.silu(
            F.layer_norm(
                self.linear_layer_2(x),
                [
                    300,
                ],
            )
        )
        return self.linear_layer_3(x), {}


def train(domain, task, seed=0):
    env = suite.load(
        domain,
        task,
        task_kwargs=dict(random=seed),
        environment_kwargs=dict(flat_observation=True),
    )

    dm_env = DMEnv(env, camera_id="cam0")  # "side" for walker
    env = wrap_env(dm_env)

    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory_size = 1000000
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, device)
    models["target_policy"] = Actor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = TD3_DEFAULT_CONFIG.copy()
    cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
    cfg["smooth_regularization_clip"] = 0.5
    cfg["discount_factor"] = 0.98
    cfg["batch_size"] = 100
    cfg["random_timesteps"] = 1000
    cfg["learning_starts"] = 1000
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 1000
    cfg["experiment"]["checkpoint_interval"] = 10000
    cfg["experiment"]["directory"] = "runs/dmc/" + domain + "/" + task

    agent = TD3(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 5000000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--domain", default="walker")
    parser.add_argument("-t", "--task", default="run")

    args = parser.parse_args()
    train(args.domain, args.task)
