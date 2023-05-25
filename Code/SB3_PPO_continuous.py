import gym
import CustomEnv
import time
import os

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn

logdir = "logs"
#models_dir = "models/DQN"
models_dir = "models/PPO"

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            #print(key)
            if key == "image":
                # We assume CxHxW images (channels first)
                # Re-ordering will be done by pre-preprocessing or wrapper
                
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                # Compute shape by doing one forward pass
                with th.no_grad():
                    ex_shape = extractors[key](th.as_tensor(observation_space.sample()[key]).float())
                    
                linear = nn.Sequential(nn.Linear(ex_shape.shape[0] * ex_shape.shape[1], 256, nn.ReLU()))  #256 is img features dim
                extractors[key] = nn.Sequential(extractors[key], linear)
                total_concat_size += 256
            
            elif key == "velocity":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16),
                    nn.ReLU()
                )
                total_concat_size += 16
            
            elif key == "relative_distance":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16),
                    nn.ReLU()
                )
                total_concat_size += 16
            
            elif key == "prev_relative_distance":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 16),
                    nn.ReLU()
                )
                total_concat_size += 16
            
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
    
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
           # if key in ["image"]:
           #     observations[key] = observations[key].permute((0, 3, 1, 2))

            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)

policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
        )

env = DummyVecEnv([lambda: Monitor(gym.make("airsim-drone-continuous-v0"))])
env = VecTransposeImage(env)

env.action_space

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=0.0001,
    verbose=1,
    batch_size=128,
    max_grad_norm=0.5,
    clip_range=0.10,
    device="cuda",
    policy_kwargs=policy_kwargs,
    tensorboard_log=logdir,
    
)

#model = PPO.load(r"C:\Users\User\Desktop\ThesisUnReal\CheckPoints\PPO\CheckPoint\rl_model_21000_steps.zip", env=env, tensorboard_log=logdir)

callbacks = []

checkpoint_callback = CheckpointCallback(
  save_freq=7_000,
  save_path="./CheckPoints/PPO/CheckPoint/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10, 
    min_evals=150, 
    verbose=1
)

eval_callback = EvalCallback(
    env,
    callback_on_new_best=True,
    n_eval_episodes=5,
    callback_after_eval=stop_train_callback,
    best_model_save_path="./CheckPoints/PPO/BestModel/",
    log_path="./CheckPoints/PPO/BestModel/",
    eval_freq=500,
)

callbacks.append(eval_callback)
callbacks.append(checkpoint_callback)

kwargs = {}
kwargs["callback"] = callbacks

TIMESTEPS = 50_000

model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO",
    **kwargs
    )
    
model.save(f"{models_dir}/{TIMESTEPS}")