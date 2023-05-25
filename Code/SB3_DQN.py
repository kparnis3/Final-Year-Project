import gym
import CustomEnv
import time
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
from stable_baselines3.common.env_checker import check_env

logdir = "logs"
#models_dir = "models/DQN"
models_dir = "models/DQN"

if not os.path.exists(models_dir): #create directories if they dont already exist
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

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

            elif key == "action_history":
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

env = DummyVecEnv([lambda: Monitor(gym.make("airsim-drone-dynamic-v0"))])
env = VecTransposeImage(env)

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=0.0001, #0.00025
    verbose=1,
    batch_size=64, #128  #32
    train_freq=4, #4
    target_update_interval=5_000, #5_000
    learning_starts=1000 , #3_000, #5_000
    policy_kwargs=policy_kwargs,
    buffer_size=50_000, #500_000
    max_grad_norm=10,
    exploration_fraction=0.6, #0.1
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log=logdir,
    double_q=True
)

#model = DQN.load(r"C:\Users\User\Desktop\ThesisUnReal\CheckPoints\DQN\CheckPoint\rl_model_14000_steps.zip", env=env, tensorboard_log=logdir)
#replay_buffer = model.load_replay_buffer(r"C:\Users\User\Desktop\ThesisUnReal\CheckPoints\DQN\CheckPoint\rl_model_replay_buffer_14000_steps.pkl")

callbacks = []

checkpoint_callback = CheckpointCallback(
  save_freq=7_000,
  save_path="./CheckPoints/DQN/CheckPoint/",
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)

stop_train_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10, 
    min_evals=80, 
    verbose=1
)

eval_callback = EvalCallback(
    env,
    #callback_on_new_best=True,
    n_eval_episodes=5,
    callback_after_eval=stop_train_callback,
    best_model_save_path="./CheckPoints/DQN/BestModel/",
    log_path="./CheckPoints/DQN/BestModel/",
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
    tb_log_name="DQN",
    **kwargs
    )
    
model.save(f"{models_dir}/{TIMESTEPS}")



#env.disconnect()
