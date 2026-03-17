from gymnasium.envs.registration import register
from sim_environment.centipede import Centipede


#Adapted from https://github.com/malteschilling/ddrl
# Register Gym environment.
register(
    id='Quant',
    entry_point='sim_environment.centipede:Centipede',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
print("Base environments registered!")
