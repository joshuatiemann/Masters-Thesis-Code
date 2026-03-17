from gymnasium.envs.registration import register
from sim_environment.ant import QuantrupedEnv, Ant

#Adapted from https://github.com/malteschilling/ddrl
# Register Gym environment.
register(
    id='Quant',
    entry_point='sim_environment.ant:QuantrupedEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='hubert',
    entry_point='sim_environment.ant:Ant',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

print("Base environments registered!")