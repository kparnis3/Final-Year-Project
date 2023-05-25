from gym.envs.registration import register

register(
    id="airsim-drone-v0",
    entry_point="CustomEnv.envs:drone_env"
)

register(
    id="airsim-drone-dynamic-v0",
    entry_point="CustomEnv.envs:drone_env_dynamic"
)

register(
    id="airsim-drone-continuous-v0",
    entry_point="CustomEnv.envs:drone_env_continuous"
)

register(
    id="airsim-drone-continuous-dynamic-v0",
    entry_point="CustomEnv.envs:drone_env_continuous_dynamic"
)